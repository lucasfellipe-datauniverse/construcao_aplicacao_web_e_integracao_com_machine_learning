#%%
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#%%
# Cria a app
app = Flask(__name__)

# Carrega o modelo e o padronizador
modelo_carregado = pickle.load(open('ml_salary.pkl', 'rb'))

# Rota para a raiz
@app.route('/')
def home():
    return render_template('home.html')

# Rota para a API de previsao
@app.route('/predict', methods=['POST'])
def predict():
    try:
         data = {
            'Country': request.form['Country'],
            'education': request.form['education'],
            'devtype' : request.form['devtype'],
            'experience': float(request.form['experience']),
            
    }
    except KeyError as e:
        return render_template("home.html", prediction_text=f"Entrada inválida. Erro: {e}")

    # Verifica se algum campo esta vazio
    if any(value == '' for value in data.values()):
        return render_template("home.html", prediction_text="Verifique se todos os campos estão preenchidos.")
    
    # Previsão com o modelo
    output = modelo_carregado.predict(dados_padronizados)[0]

    # Formata a saida
    formatted_output = round(output, 2)
    
    # Renderiza o html com a previsão do modelo
    return render_template("home.html", prediction_text="$ {} [valor anual]".format(formatted_output))

# Executa a app
if __name__ == "__main__":
    app.run()
