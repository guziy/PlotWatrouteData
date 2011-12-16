__author__ = 'huziy'

from ccc.ccc import champ_ccc

def get_bedrock_depth_amno_180x172():
    """
    Returns 2d field of the bedrock depth in meters
    in amno domain
    """
    path = "data/ccc_data/dpth_180x172"
    file_obj = champ_ccc(fichier=path)
    the_field = file_obj.charge_champs()[0]["field"]
    return the_field
    pass