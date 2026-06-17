# Dashboard de Absenteísmo no Trabalho

Painel interativo em **Streamlit** que apresenta um retrato do período da base
*Absenteeism at Work*: distribuição das ausências por mês, estação, dia da semana
e motivo, relações entre variáveis e os resultados do modelo de Regressão Binomial
Negativa.

## Como executar
```bash
pip install -r requirements.txt
streamlit run app.py
```
Acesse `http://localhost:8501`.

## Arquivos
- `app.py` — aplicação Streamlit (3 abas + filtros).
- `absenteeism_clean.csv` — base tratada (697 registros) usada pelo painel.
- `requirements.txt` — dependências.
