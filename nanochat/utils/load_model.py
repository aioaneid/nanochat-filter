def remap_attention_keys(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if ".attn.c_q." in k:
            k = k.replace(".attn.c_q.", ".attn.qkv_computer.c_q.")
        elif ".attn.c_k." in k:
            k = k.replace(".attn.c_k.", ".attn.qkv_computer.c_k.")
        elif ".attn.c_v." in k:
            k = k.replace(".attn.c_v.", ".attn.qkv_computer.c_v.")
        new_sd[k] = v
    return new_sd
