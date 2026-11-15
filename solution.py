"""Regressão linear múltipla para os dados de imóveis Lar Perfeito.

O modelo estimado é:
    preco_venda = beta_0 + beta_1 * area_util_m2 + beta_2 * quantidade_quartos

A função principal recebe o caminho de um CSV e retorna [beta_0, beta_1, beta_2].
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd


ColunaPath = Union[str, Path]


def regressao_linear(arquivo: ColunaPath) -> List[float]:
    """Ajusta a regressão linear e retorna [intercepto, beta_area, beta_quartos].

    Args:
        arquivo: caminho para o CSV com as colunas `area_util_m2`,
            `quantidade_quartos` e `preco_venda`.

    Returns:
        Lista de três coeficientes float, nesta ordem:
        [beta_0, beta_1, beta_2].

    Raises:
        ValueError: se uma coluna necessária estiver ausente ou contiver
            valores ausentes/não numéricos.
    """
    df = pd.read_csv(arquivo)
    colunas = ["area_util_m2", "quantidade_quartos", "preco_venda"]
    ausentes = [coluna for coluna in colunas if coluna not in df.columns]
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {ausentes}")

    dados = df[colunas].apply(pd.to_numeric, errors="coerce")
    if dados.isna().any().any():
        raise ValueError("As colunas do modelo contêm valores ausentes ou não numéricos.")
    if dados.empty:
        raise ValueError("O CSV não contém linhas de dados para ajustar o modelo.")

    # Matriz de projeto X = [1, área, quartos]. A primeira coluna de 1's
    # representa o intercepto beta_0, como nos slides da disciplina.
    X = np.column_stack(
        (
            np.ones(len(dados), dtype=float),
            dados["area_util_m2"].to_numpy(dtype=float),
            dados["quantidade_quartos"].to_numpy(dtype=float),
        )
    )
    y = dados["preco_venda"].to_numpy(dtype=float)
    # Fórmula matricial dos mínimos quadrados: beta = (XᵀX)⁻¹Xᵀy.
    Xt = np.transpose(X)
    coeficientes = np.linalg.inv(Xt @ X) @ Xt @ y
    return [float(coeficiente) for coeficiente in coeficientes]


# Nome alternativo descritivo, caso o restante do projeto use essa convenção.
calcular_coeficientes = regressao_linear


if __name__ == "__main__":
    caminho_csv = Path(__file__).with_name("imoveis_lar_perfeito.csv")
    b0, b1, b2 = regressao_linear(caminho_csv)
    print("Coeficientes [beta_0, beta_1, beta_2]:", [b0, b1, b2])
    print(
        "Equação: preco_venda = "
        f"{b0:.6f} + ({b1:.6f} * area_util_m2) "
        f"+ ({b2:.6f} * quantidade_quartos)"
    )
