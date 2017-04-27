import string
import numpy as np
GOOD_CHARS = string.ascii_lowercase+" ,.;'-\"\n"
CHARS_LEN = len(GOOD_CHARS)

def nice_string(raw_str):
    s = (raw_str.replace("\n\n","\0")
                .replace("\n"," ")
                .replace("\0","\n")
                .replace("”",'"')
                .replace("“",'"')
                .replace("’","'"))
    return "".join(c.lower() for c in s if c.lower() in GOOD_CHARS)
def char_to_vec(c):
    pos = GOOD_CHARS.index(c)
    vec = np.zeros(CHARS_LEN,dtype="float32")
    vec[pos] = 1.0
    return vec
def in_vec(s):
    return [char_to_vec(c) for c in s]
def get_char(vec):
    ls = list(vec)
    idx = ls.index(max(ls))
    return GOOD_CHARS[idx]
def get_str(filename):
    return nice_string(get_raw_str(filename))
def get_raw_str(filename):
    with open(filename,encoding="utf8") as file:
        return file.read()
def out_list_to_str(outlist):
    return "".join(get_char(v) for v in outlist)
def calc_str_errors(output_str,correct_str):
    assert len(output_str) == len(correct_str)
    return len(output_str) - sum(o==c for o,c in zip(output_str,correct_str))
