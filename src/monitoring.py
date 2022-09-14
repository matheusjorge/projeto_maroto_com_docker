from scipy.stats import ks_2samp
def ks_test(sample1, sample2, alpha=0.05):
    """
    Calcula o KS para duas variáveis aleatórias.

    """
    ks_stat, p_value = ks_2samp(sample1, sample2)
    assert p_value >= alpha, "Possível mudança de dados"
    return ks_stat, p_value