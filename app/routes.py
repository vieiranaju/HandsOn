from flask import render_template, request, redirect, url_for
from app import app
from app.utils import salvar_usuarios, carregar_usuarios, enviar_email
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        nome = request.form['nome']
        sobrenome = request.form['sobrenome']
        email = request.form['email']
        telefone = request.form['telefone']
        nascimento = f"{request.form['dia']}/{request.form['mes']}/{request.form['ano']}"
        
        usuario = {
            'nome': nome,
            'sobrenome': sobrenome,
            'email': email,
            'telefone': telefone,
            'nascimento': nascimento
        }
        
        usuarios = carregar_usuarios()
        usuarios.append(usuario)
        salvar_usuarios(usuarios)
        
        return redirect(url_for('cadastro_sucesso'))
    
    return render_template('cadastro.html')

@app.route('/cadastro_sucesso')
def cadastro_sucesso():
    return 'Cadastro realizado com sucesso!'

@app.route('/prever', methods=['GET'])
def prever():
    # Carregar dados do JSON
    with open('dados/BTCUSDT_1D.json') as f:
        data = json.load(f)

    # Extrair preços de fechamento e datas
    close_prices = [float(item[4]) for item in data]
    dates = [pd.to_datetime(item[0], unit='ms') for item in data]

    # Converter para DataFrame
    dataframe = pd.DataFrame(close_prices, columns=['Close'], index=dates)
    dataset = dataframe.values.astype('float32')

    # Normalizar o dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Dividir em conjuntos de treino e teste
    train_size = int(len(dataset) * 0.67)
    train, test = dataset[:train_size], dataset[train_size:]

    # Criar datasets X e Y
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Redimensionar entrada para [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Criar e ajustar a rede LSTM
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # Prever próximos 3 dias
    input_seq = dataset[-look_back:].reshape((1, 1, look_back))
    future_predictions = []
    for _ in range(3):
        next_pred = model.predict(input_seq)
        future_predictions.append(next_pred[0, 0])
        input_seq = np.append(input_seq[:, :, 1:], next_pred.reshape(1, 1, 1), axis=2)

    # Inverter previsões futuras
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Gerar datas para previsões futuras
    future_dates = pd.date_range(start=dataframe.index[-1], periods=4, inclusive='right').to_pydatetime().tolist()

    # Criar o gráfico animado
    fig, ax = plt.subplots()
    ax.plot(dates, scaler.inverse_transform(dataset), label='Original Data')
    line, = ax.plot([], [], 'r-', label='Prediction')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        xdata = np.concatenate((dates, future_dates[:frame + 1]))
        ydata = np.concatenate((scaler.inverse_transform(dataset).flatten(), future_predictions[:frame + 1].flatten()))
        line.set_data(xdata, ydata)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(3), init_func=init, blit=True, repeat=False)
    plt.legend()

    # Salvar a animação como GIF em memória
    gif_buffer = BytesIO()
    ani.save(gif_buffer, format='gif', writer='imagemagick', fps=1)
    gif_buffer.seek(0)

    # Enviar o email com o GIF anexado
    usuarios = carregar_usuarios()
    for usuario in usuarios:
        enviar_email(usuario['email'], 'Previsão de Preços', 'Veja a previsão dos próximos 3 dias no gráfico anexado.', gif_buffer)

    return 'Previsões geradas e emails enviados com sucesso!'
