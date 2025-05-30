import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from .parsers import parse_a3m, parse_fasta, parse_mixed_fasta, read_template_pdb, parse_pdb_w_seq, read_templates
from .RoseTTAFoldModel  import RoseTTAFoldModule
from .util import *
from collections import namedtuple
from .ffindex import *
from .data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from .kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d
from .util_module import XYZConverter
from .chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS, INIT_NA_CRDS

# suppress dgl warning w/ newest pytorch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define amino acid and nucleotide mappings
AA_TO_NUM = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    '-': 20, '.': 20  # gap characters
}

NT_TO_NUM = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3,
    '-': 4, '.': 4  # gap characters
}

def dna_reverse_complement_str(seq):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join([complement.get(base, base) for base in reversed(seq)])

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold2NA")
    parser.add_argument("-protein_seq", help="Protein sequence", type=str, required=True)
    parser.add_argument("-dna_seq", help="DNA sequence", type=str, required=True)
    parser.add_argument("-db", help="HHpred database location", default=None)
    parser.add_argument("-model", default=None, help="The model weights", required=True)
    args = parser.parse_args()
    return args

MAX_CYCLE = 10
NMODELS = 1
NBIN = [37, 37, 37, 19]

MAXLAT = 256
MAXSEQ = 2048

MODEL_PARAM ={
        "n_extra_block"   : 4,
        "n_main_block"    : 32,
        "n_ref_block"     : 4,
        "d_msa"           : 256 ,
        "d_pair"          : 128,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 4,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 64,
        "p_drop"       : 0.0,
        "lj_lin"       : 0.75
}

SE3_param = {
        "num_layers"    : 1,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
}

SE3_ref_param = {
        "num_layers"    : 2,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
}

MODEL_PARAM['SE3_param_full'] = SE3_param
MODEL_PARAM['SE3_param_topk'] = SE3_ref_param

def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    
    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

def pae_unbin(pred_pae):
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)

class Predictor():
    def __init__(self, model_weights, device="cpu"):
        self.model_weights = model_weights
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        self.model = RoseTTAFoldModule(
            **MODEL_PARAM,
            aamask=allatom_mask.to(self.device),
            ljlk_parameters=ljlk_parameters.to(self.device),
            lj_correction_parameters=lj_correction_parameters.to(self.device),
            num_bonds=num_bonds.to(self.device),
            hbtypes=hbtypes.to(self.device),
            hbbaseatoms=hbbaseatoms.to(self.device),
            hbpolys=hbpolys.to(self.device)
        ).to(self.device)

        could_load = self.load_model(self.model_weights)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

        self.xyz_converter = XYZConverter()

    def load_model(self, model_weights):
        if not os.path.exists(model_weights):
            return False
        checkpoint = torch.load(model_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True

    def predict(self, protein_seq, dna_seq, n_templ=4):
        # Create fake MSA with just the input sequences
        protein_seq = protein_seq.upper()
        dna_seq = dna_seq.upper()
        
        # Create protein MSA (just 1 sequence)
        protein_msa = np.array([[AA_TO_NUM[a] for a in protein_seq]])
        protein_ins = np.zeros_like(protein_msa)
        
        # Create DNA MSA (just 1 sequence)
        dna_msa = np.array([[NT_TO_NUM[a] for a in dna_seq]])
        dna_ins = np.zeros_like(dna_msa)
        
        # Create reverse complement of DNA
        dna_msa_rc = np.array([[NT_TO_NUM[a] for a in dna_reverse_complement_str(dna_seq)]])
        
        # Combine all sequences
        Ls = [len(protein_seq), len(dna_seq), len(dna_seq)]  # protein, dna, dna_rc
        msa_orig = {'msa':torch.tensor(protein_msa), 'ins':torch.tensor(protein_ins)}
        msa_orig = merge_a3m_hetero(msa_orig, {'msa':torch.tensor(dna_msa), 'ins':torch.tensor(dna_ins)}, [Ls[0], Ls[1]])
        msa_orig = merge_a3m_hetero(msa_orig, {'msa':torch.tensor(dna_msa_rc), 'ins':torch.tensor(dna_ins)}, [sum(Ls[:2]), Ls[2]])
        
        msa_orig, ins_orig = msa_orig['msa'], msa_orig['ins']

        # Process templates (using empty templates since we don't have any)
        L = sum(Ls)
        xyz_t = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(n_templ,L,1,1) + torch.rand(n_templ,L,1,3)*5.0 - 2.5
        is_NA = torch.zeros(L, dtype=torch.bool)
        is_NA[Ls[0]:] = True  # Mark DNA positions
        xyz_t[:,is_NA] = INIT_NA_CRDS.reshape(1,1,NTOTAL,3)

        mask_t = torch.full((n_templ, L, NTOTAL), False) 
        t1d = torch.nn.functional.one_hot(torch.full((n_templ, L), 20).long(), num_classes=NAATOKENS-1).float()
        t1d = torch.cat((t1d, torch.zeros((n_templ,L,1)).float()), -1)

        same_chain = torch.zeros((1,L,L), dtype=torch.bool, device=xyz_t.device)
        stopres = 0
        for i in range(1,len(Ls)):
            startres,stopres = sum(Ls[:(i-1)]), sum(Ls[:i])
            same_chain[:,startres:stopres,startres:stopres] = True
        same_chain[:,stopres:,stopres:] = True

        # template features
        xyz_t = xyz_t.float().unsqueeze(0)
        mask_t = mask_t.unsqueeze(0)
        t1d = t1d.float().unsqueeze(0)

        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None]
        mask_t_2d = mask_t_2d.float()*same_chain.float()[:,None]
        t2d = xyz_to_t2d(xyz_t, mask_t_2d)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,L,NTOTAL,3), seq_tmp, mask_in=mask_t.reshape(-1,L,NTOTAL))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))

        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 3*NTOTALDOFS)

        self.model.eval()
        with torch.no_grad():
            seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
                msa_orig, ins_orig, p_mask=0.0, params={'MAXLAT': MAXLAT, 'MAXSEQ': MAXSEQ, 'MAXCYCLE': MAX_CYCLE})

            _, N, L = msa_seed.shape[:3]
            B = 1   
            
            idx_pdb = torch.arange(L).long().view(1, L)
            for i in range(len(Ls)-1):
                idx_pdb[ :, sum(Ls[:(i+1)]): ] += 100

            seq = seq.unsqueeze(0)
            msa_seed = msa_seed.unsqueeze(0)
            msa_extra = msa_extra.unsqueeze(0)

            t1d = t1d.to(self.device)
            t2d = t2d.to(self.device)
            idx_pdb = idx_pdb.to(self.device)
            xyz_t = xyz_t.to(self.device)
            alpha_t = alpha_t.to(self.device)
            xyz = xyz_t[:,0].to(self.device)
            same_chain = same_chain.to(self.device)
            mask_t_2d = mask_t_2d.to(self.device)

            msa_prev = None
            pair_prev = None
            alpha_prev = torch.zeros((1,L,NTOTALDOFS,2), device=self.device)
            xyz_prev=xyz
            state_prev = None

            best_lddt = torch.tensor([-1.0], device=self.device)
            best_pae = None
            
            for i_cycle in range(MAX_CYCLE):
                msa_seed_i = msa_seed[:,i_cycle].to(self.device)
                msa_extra_i = msa_extra[:,i_cycle].to(self.device)
                seq_i = seq[:,i_cycle].to(self.device)
                
                with torch.cuda.amp.autocast(True):
                    logit_s, logit_aa_s, logit_pae, p_bind, init_crds, alpha_prev, _, pred_lddt_binned, msa_prev, pair_prev, state_prev = self.model(
                        msa_latent=msa_seed_i, 
                        msa_full=msa_extra_i,
                        seq=seq_i, 
                        seq_unmasked=seq_i, 
                        xyz=xyz_prev, 
                        sctors=alpha_prev,
                        idx=idx_pdb,
                        t1d=t1d, 
                        t2d=t2d,
                        xyz_t=xyz_t[:,:,:,1],
                        mask_t=mask_t_2d,
                        alpha_t=alpha_t,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        same_chain=same_chain
                    )

                xyz_prev = init_crds[-1]
                alpha_prev = alpha_prev[-1]
                pred_lddt = lddt_unbin(pred_lddt_binned)
                pae = pae_unbin(logit_pae)

                if pred_lddt.mean() > best_lddt.mean():
                    best_lddt = pred_lddt.clone()
                    best_pae = pae.clone()

            # Calculate average pLDDT and PAE for the protein-DNA interface
            # Get protein and DNA residue indices
            protein_indices = range(0, Ls[0])
            dna_indices = range(Ls[0], Ls[0]+Ls[1])
            
            # Calculate interface pLDDT (average of protein and DNA)
            protein_lddt = best_lddt[0, protein_indices].mean()
            dna_lddt = best_lddt[0, dna_indices].mean()
            avg_lddt = (protein_lddt + dna_lddt) / 2
            
            # Calculate interface PAE (between protein and DNA)
            interface_pae = best_pae[0, protein_indices, :][:, dna_indices].mean()
            
            # Determine binary label
            label = 1 if avg_lddt > 0.8 else 0
            
            return label, avg_lddt.item(), interface_pae.item()

if torch.cuda.is_available():
    print("Running on GPU")
    predictorBase = Predictor("./weights/RF2NA_apr23.pt", torch.device("cuda:0"))
else:
    print("Running on CPU")
    predictorBase = Predictor("./weights/RF2NA_apr23.pt", torch.device("cpu"))

# if __name__ == "__main__":
#     args = get_args()
#
#     if torch.cuda.is_available():
#         print("Running on GPU")
#         pred = Predictor(args.model, torch.device("cuda:0"))
#     else:
#         print("Running on CPU")
#         pred = Predictor(args.model, torch.device("cpu"))
#
#     label, avg_lddt, interface_pae = pred.predict(
#         protein_seq=args.protein_seq,
#         dna_seq=args.dna_seq,
#     )
#
#     print(f"Predicted label: {label}")
#     print(f"Average pLDDT: {avg_lddt:.2f}")