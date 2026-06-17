# -*- coding: utf-8 -*-
"""
Trabalho A4 - Modelos Lineares Generalizados
Dashboard de Absenteismo no Trabalho
Leonardo Araujo Pereira - Ciencia de Dados e Inteligencia Artificial - IESB

Painel interativo que apresenta um RETRATO FIEL DO PERIODO DA BASE
(distribuicao das ausencias por mes, estacao, dia da semana e motivo),
alem das relacoes estudadas na Parte 2 e do modelo Binomial Negativa.

Execucao:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Absenteismo no Trabalho - MLG",
                   page_icon="📊", layout="wide")

RED = "#AA0000"
SEQ = ["#AA0000", "#cc4444", "#e08585", "#f0bcbc"]

MES = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
       7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
SEASON = {1:"Verao",2:"Outono",3:"Inverno",4:"Primavera"}
DIA = {2:"Segunda",3:"Terca",4:"Quarta",5:"Quinta",6:"Sexta"}
EDU = {1:"Ensino Medio",2:"Graduacao",3:"Pos-graduacao",4:"Mestrado/Doutorado"}


@st.cache_data
def load_data():
    """Carrega base tratada; se ausente, trata a partir do CSV bruto."""
    here = os.path.dirname(__file__)
    clean = os.path.join(here, "absenteeism_clean.csv")
    if os.path.exists(clean):
        df = pd.read_csv(clean)
    else:
        raw = os.path.join(here, "..", "ABSENTEEISM.csv")
        df = pd.read_csv(raw, sep=";", decimal=",")
        df.columns = [c.strip() for c in df.columns]
        df = df[(df["Reason_for_absence"] != 0) & (df["Month_of_absence"] != 0)].copy()
        df["Reason_group"] = np.where(df["Reason_for_absence"] <= 21,
                                      "Doenca_CID", "Consulta_Exame")
    df["Mes"] = df["Month_of_absence"].map(MES)
    df["Estacao"] = df["Seasons"].map(SEASON)
    df["Dia"] = df["Day_of_the_week"].map(DIA)
    df["Escolaridade"] = df["Education"].map(EDU)
    df["Motivo"] = df["Reason_group"].map(
        {"Doenca_CID": "Doenca (CID-10)", "Consulta_Exame": "Consulta/Exame"})
    df["Bebe"] = df["Social_drinker"].map({0: "Nao", 1: "Sim"})
    return df


df = load_data()

# ----------------------------------------------------------------------
# Cabecalho
# ----------------------------------------------------------------------
st.title("📊 Absenteismo no Trabalho — Painel Analitico")
st.caption("Modelos Lineares Generalizados • Leonardo Araujo Pereira • "
           "Ciencia de Dados e IA — IESB")

# ----------------------------------------------------------------------
# Filtros (sidebar)
# ----------------------------------------------------------------------
st.sidebar.header("Filtros do periodo")
estacoes = st.sidebar.multiselect("Estacao", list(SEASON.values()),
                                  default=list(SEASON.values()))
motivos = st.sidebar.multiselect("Motivo", df["Motivo"].unique().tolist(),
                                 default=df["Motivo"].unique().tolist())
meses_sel = st.sidebar.slider("Intervalo de meses", 1, 12, (1, 12))

mask = (df["Estacao"].isin(estacoes) & df["Motivo"].isin(motivos) &
        df["Month_of_absence"].between(meses_sel[0], meses_sel[1]))
d = df[mask]

st.sidebar.markdown("---")
st.sidebar.metric("Registros filtrados", f"{len(d)}")
st.sidebar.caption("Base tratada: 697 registros de 36 funcionarios, "
                   "apos remocao de zeros estruturais (motivo 0 / mes 0).")

# ----------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros de ausencia", f"{len(d)}")
c2.metric("Total de horas", f"{int(d['Absenteeism_time_in_hours'].sum()):,}".replace(",", "."))
c3.metric("Media de horas/ausencia", f"{d['Absenteeism_time_in_hours'].mean():.1f}")
c4.metric("Funcionarios distintos", f"{d['ID'].nunique()}")

st.markdown("---")

# ----------------------------------------------------------------------
# Abas
# ----------------------------------------------------------------------
aba1, aba2, aba3 = st.tabs(["🗓️ Retrato do periodo",
                            "🔗 Relacoes entre variaveis",
                            "🤖 Modelo (Binomial Negativa)"])

# ------------------------- ABA 1: PERIODO -----------------------------
with aba1:
    st.subheader("Retrato fiel do periodo da base")
    col1, col2 = st.columns(2)

    mes = (d.groupby("Month_of_absence")["Absenteeism_time_in_hours"]
             .sum().reindex(range(1, 13), fill_value=0))
    fig = px.bar(x=[MES[i] for i in mes.index], y=mes.values,
                 labels={"x": "Mes", "y": "Horas de ausencia"},
                 title="Horas de ausencia por mes", color_discrete_sequence=[RED])
    col1.plotly_chart(fig, use_container_width=True)

    sea = d.groupby("Estacao")["Absenteeism_time_in_hours"].sum()
    fig = px.pie(values=sea.values, names=sea.index, hole=.45,
                 title="Distribuicao das horas por estacao",
                 color_discrete_sequence=SEQ)
    col2.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    dia = (d.groupby("Day_of_the_week")["Absenteeism_time_in_hours"]
             .mean().reindex([2, 3, 4, 5, 6]))
    fig = px.bar(x=[DIA[i] for i in dia.index], y=dia.values,
                 labels={"x": "Dia da semana", "y": "Media de horas"},
                 title="Media de horas por dia da semana",
                 color_discrete_sequence=[RED])
    col3.plotly_chart(fig, use_container_width=True)

    mot = d.groupby("Motivo")["Absenteeism_time_in_hours"].sum()
    fig = px.bar(x=mot.index, y=mot.values,
                 labels={"x": "Motivo", "y": "Horas de ausencia"},
                 title="Horas de ausencia por grupo de motivo",
                 color_discrete_sequence=[RED])
    col4.plotly_chart(fig, use_container_width=True)

    st.info("A base nao possui ano-calendario; o periodo e descrito pelos campos "
            "temporais Mes, Estacao e Dia da semana — que compoem o retrato fiel "
            "da sazonalidade do absenteismo.")

# ------------------------- ABA 2: RELACOES ----------------------------
with aba2:
    st.subheader("Correlacao, associacao e relacao entre variaveis")
    quant = ["Absenteeism_time_in_hours", "Transportation_expense",
             "Distance_from_Residence_to_Work", "Service_time", "Age",
             "Work_load_Average_day", "Hit_target", "Son", "Pet",
             "Weight", "Height", "Body_mass_index"]
    corr = d[quant].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Matriz de correlacao (quantitativas)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    fig = px.box(d, x="Motivo", y="Absenteeism_time_in_hours",
                 color="Motivo", color_discrete_sequence=SEQ,
                 title="Horas por motivo (quantitativa x qualitativa)")
    col1.plotly_chart(fig, use_container_width=True)

    fig = px.box(d, x="Bebe", y="Absenteeism_time_in_hours",
                 color="Bebe", color_discrete_sequence=SEQ,
                 title="Horas por consumo social de alcool")
    col2.plotly_chart(fig, use_container_width=True)

# ------------------------- ABA 3: MODELO ------------------------------
with aba3:
    st.subheader("Modelo de Regressao Binomial Negativa (GLM)")
    st.markdown(
        "O alvo **horas de ausencia** e uma contagem com forte sobredispersao "
        "(variancia/media ≈ 25,7), o que torna a **Regressao Binomial Negativa** "
        "mais adequada que Poisson e que a regressao linear. Abaixo, as razoes de "
        "taxa (IRR = exp(β)) estimadas no SAS (PROC GENMOD), validadas em Python.")

    irr = pd.DataFrame({
        "Variavel": ["Motivo: Doenca (CID) vs Consulta", "Bebe socialmente (sim)",
                     "Num. de filhos (+1)", "Idade (+1 ano)",
                     "Quinta-feira vs Segunda", "Mestrado/Doutorado vs Ens. Medio"],
        "IRR": [3.18, 1.46, 1.14, 1.025, 0.71, 0.32],
        "p-valor": [0.0000, 0.0002, 0.0014, 0.0038, 0.0056, 0.0271],
    })
    fig = px.bar(irr.sort_values("IRR"), x="IRR", y="Variavel", orientation="h",
                 color="IRR", color_continuous_scale="RdBu_r",
                 title="Razoes de taxa (IRR) — efeito sobre as horas de ausencia")
    fig.add_vline(x=1, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(irr, use_container_width=True, hide_index=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AIC (NB)", "2770", delta="-2090 vs Poisson", delta_color="inverse")
    m2.metric("Pseudo R² (CS)", "0,56")
    m3.metric("MAE (teste)", "5,1 h")
    m4.metric("RMSE (teste)", "12,1 h")
    st.caption("IRR > 1 aumenta a taxa esperada de horas; IRR < 1 reduz. "
               "Ex.: ausencias por doenca (CID) tem taxa 3,2x maior que consultas/exames.")
