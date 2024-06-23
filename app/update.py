import json
import requests

def ler_ultimo_registro(nome_arquivo):
    with open(nome_arquivo, 'r') as file:
        dados = json.load(file)
        if dados:
            ultimo_registro = dados[-1][0]
            return ultimo_registro
        else:
            return None

def obter_dados_binance(symbol, start_time, interval, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&startTime={start_time}&limit={limit}&interval={interval}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Erro ao chamar a API da Binance")
        return []

def atualizar_arquivo(nome_arquivo, symbol, interval):
    # Obtém o último registro do arquivo
    ultimo_timestamp = ler_ultimo_registro(nome_arquivo)

    # Se houver um último registro, faz uma chamada à API usando esse timestamp como startTime
    if ultimo_timestamp:
        dados_novos = obter_dados_binance(symbol, ultimo_timestamp+1, interval)
        if dados_novos:
            # Atualiza o arquivo com os novos dados
            with open(nome_arquivo, 'r+') as file:
                dados_existentes = json.load(file)
                dados_existentes.extend(dados_novos)
                file.seek(0)
                json.dump(dados_existentes, file, indent=4)
                file.truncate()
            print("Arquivo atualizado com sucesso!")
        else:
            print("Nenhum dado novo encontrado.")
    else:
        print("Não foi possível ler o último registro do arquivo.")

# Defina o nome do arquivo, o símbolo e o intervalo conforme necessário
nome_arquivo = 'dados/BTCUSDT_1D.json'
symbol = 'BTCUSDT'
interval = '1d'

# Chama a função para atualizar o arquivo com os novos dados da Binance
atualizar_arquivo(nome_arquivo, symbol, interval)

nome_arquivo = 'dados/BTCUSDT_4H.json'
interval = '4h'
atualizar_arquivo(nome_arquivo, symbol, interval)

nome_arquivo = 'dados/BTCUSDT_1H.json'
interval = '1h'
atualizar_arquivo(nome_arquivo, symbol, interval)
