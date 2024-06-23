import smtplib
from email.message import EmailMessage
from email.mime.image import MIMEImage
import json

def enviar_email(destinatario, assunto, corpo, anexo=None):
    EMAIL_ADDRESS = 'seu_email@example.com'
    EMAIL_PASSWORD = 'sua_senha'

    msg = EmailMessage()
    msg['Subject'] = assunto
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = destinatario
    msg.set_content(corpo)

    if anexo:
        anexo.seek(0)
        mime_image = MIMEImage(anexo.read(), _subtype="gif")
        mime_image.add_header('Content-Disposition', 'attachment', filename='forecast_animation.gif')
        msg.attach(mime_image)

    with smtplib.SMTP_SSL('smtp.example.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

def carregar_usuarios(filename='usuarios.json'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def salvar_usuarios(usuarios, filename='usuarios.json'):
    with open(filename, 'w') as file:
        json.dump(usuarios, file)
