# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import re

import esm
import torch
from argparse import Namespace
import warnings
import urllib
from pathlib import Path
import sys

from esm.modelv2.esm2 import ESM2

from .json_file import *


def load_model_and_alphabet(model_name, max_tokens_per_msa=2 ** 16):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name, max_tokens_per_msa=max_tokens_per_msa)
    else:
        return load_model_and_alphabet_hub(model_name)


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location='cpu')
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    return data


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v and ESM-IF"""
    return not ("esm1v" in model_name or "esm_if" in model_name)


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def load_model_and_alphabet_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    regression_data = load_regression_hub(model_name)
    return load_model_and_alphabet_core(model_data, regression_data)


def load_model_and_alphabet_local(model_location, cuda_lst=None, map_location='cpu', regression_location=None, max_tokens_per_msa=2 ** 16):
    """ Load from local path. The regression weights need to be co-located """
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location=map_location, weights_only=False)
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        if os.path.isfile(regression_location):
            regression_data = torch.load(regression_location, map_location=map_location, weights_only=False)
        else:
            warnings.warn('miss regression weights!')
            regression_data = None
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_name, model_data, regression_data, max_tokens_per_msa=max_tokens_per_msa)

    # try:
    #     if regression_location is None:
    #         regression_location = model_location[:-3] + "-contact-regression.pt"
    #     regression_data = torch.load(regression_location, map_location=map_location)
    # except FileNotFoundError:
    #     regression_data = None
    # return load_model_and_alphabet_core(model_data, cuda_lst, regression_data)


def has_emb_layer_norm_before(model_state):
    """Determine whether layer norm needs to be applied before the encoder"""
    return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())


def _load_model_and_alphabet_core_v1(model_data, max_tokens_per_msa=2 ** 16):
    import esm  # since esm.inverse_folding is imported below, you actually have to re-import esm here

    alphabet = esm.Alphabet.from_architecture(model_data["args"].arch)

    if model_data["args"].arch == "roberta_large":
        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
        model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop
        model_args["emb_layer_norm_before"] = has_emb_layer_norm_before(model_state)
        model_type = esm.ProteinBertModel

    elif model_data["args"].arch == "protein_bert_base":

        # upgrade state dict
        pra = lambda s: "".join(s.split("decoder_")[1:] if "decoder" in s else s)
        prs = lambda s: "".join(s.split("decoder.")[1:] if "decoder" in s else s)
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
        model_type = esm.ProteinBertModel
    elif model_data["args"].arch == "msa_transformer":

        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        prs3 = lambda s: s.replace("row", "column") if "row" in s else s.replace("column", "row")
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_args['max_tokens_per_msa'] = max_tokens_per_msa
        model_state = {prs1(prs2(prs3(arg[0]))): arg[1] for arg in model_data["model"].items()}
        if model_args.get("embed_positions_msa", False):
            emb_dim = model_state["msa_position_embedding"].size(-1)
            model_args["embed_positions_msa_dim"] = emb_dim  # initial release, bug: emb_dim==1
            model_type = esm.MSATransformer1b
        else:
            model_type = esm.MSATransformer

    elif "invariant_gvp" in model_data["args"].arch:
        import esm.inverse_folding

        model_type = esm.inverse_folding.gvp_transformer.GVPTransformerModel
        model_args = vars(model_data["args"])  # convert Namespace -> dict

        def update_name(s):
            # Map the module names in checkpoints trained with internal code to
            # the updated module names in open source code
            s = s.replace("W_v", "embed_graph.embed_node")
            s = s.replace("W_e", "embed_graph.embed_edge")
            s = s.replace("embed_scores.0", "embed_confidence")
            s = s.replace("embed_score.", "embed_graph.embed_confidence.")
            s = s.replace("seq_logits_projection.", "")
            s = s.replace("embed_ingraham_features", "embed_dihedrals")
            s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
            s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
            return s

        model_state = {
            update_name(sname): svalue
            for sname, svalue in model_data["model"].items()
            if "version" not in sname
        }

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )

    return model, alphabet, model_state


def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict


def load_model_and_alphabet_core(model_name, model_data, regression_data=None, max_tokens_per_msa=2 ** 16):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    if model_name.startswith("esm2"):
        model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    else:
        model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data, max_tokens_per_msa=max_tokens_per_msa)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet


def init_model_and_alphabet(args_json, cuda_lst=None):
    """ Load from local path. The regression weights need to be co-located """
    args_dict = read_json(args_json)
    args = Namespace(**args_dict)
    args.cuda_lst = cuda_lst
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # esm.ProteinBertModel.add_args(parser)
    # args = parser.parse_args()
    alphabet = esm.Alphabet.from_architecture('roberta_large')
    model = esm.ProteinBertModel(args=args, alphabet=alphabet)
    model = init_model_and_alphabet_core(model, alphabet)
    return model, alphabet


def init_model_and_alphabet_core(model, alphabet):
    # upgrade state dict
    model_state = model.state_dict()
    model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()

    # pra = lambda s: ''.join(s.split('encoder_')[1:] if 'encoder' in s else s)
    # prs1 = lambda s: ''.join(s.split('encoder.')[1:] if 'encoder' in s else s)
    # prs2 = lambda s: ''.join(s.split('sentence_encoder.')[1:] if 'sentence_encoder' in s else s)
    # model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
    # model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
    # model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop

    expected_keys = set(model_state.keys())
    found_keys = set(model_state.keys())

    expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
    error_msgs = []
    missing = (expected_keys - found_keys) - expected_missing
    if missing:
        error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
    unexpected = found_keys - expected_keys
    if unexpected:
        error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

    if error_msgs:
        raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)))
    if expected_missing - found_keys:
        warnings.warn("Regression weights not found, predicting contacts will not produce correct results.")

    model.load_state_dict(model_state)

    return model


def esm1_t34_670M_UR50S():
    """ 34 layer transformer model with 670M params, trained on Uniref50 Sparse.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t34_670M_UR50S")


def esm1_t34_670M_UR50D():
    """ 34 layer transformer model with 670M params, trained on Uniref50 Dense.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t34_670M_UR50D")


def esm1_t34_670M_UR100():
    """ 34 layer transformer model with 670M params, trained on Uniref100.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t34_670M_UR100")


def esm1_t12_85M_UR50S():
    """ 12 layer transformer model with 85M params, trained on Uniref50 Sparse.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t12_85M_UR50S")


def esm1_t6_43M_UR50S():
    """ 6 layer transformer model with 43M params, trained on Uniref50 Sparse.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1_t6_43M_UR50S")


def esm1b_t33_650M_UR50S():
    """ 33 layer transformer model with 650M params, trained on Uniref50 Sparse.
    This is our best performing model, which will be described in a future publication.

    Returns a tuple of (Model, Alphabet).
    """
    return load_model_and_alphabet_hub("esm1b_t33_650M_UR50S")


def esm_msa1_t12_100M_UR50S():
    return load_model_and_alphabet_hub("esm_msa1_t12_100M_UR50S")
