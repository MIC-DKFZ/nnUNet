import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, sender_password, recipient_email, subject, body):
    """
    Verstuurt een e-mail via SMTP.

    Parameters:
    - sender_email (str): Het e-mailadres van de afzender.
    - sender_password (str): Het wachtwoord van de afzender.
    - recipient_email (str): Het e-mailadres van de ontvanger.
    - subject (str): Het onderwerp van de e-mail.
    - body (str): De tekst in de e-mail.

    Returns:
    - None
    """
    # Stel de e-mail samen
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Verbinden met de SMTP-server
        with smtplib.SMTP('smtp.office365.com', 587) as server:
            server.starttls()  # Versleutelde verbinding
            server.login(sender_email, sender_password)  # Inloggen
            server.send_message(msg)  # E-mail versturen
            print("E-mail succesvol verstuurd!")
    except Exception as e:
        print(f"Er ging iets mis: {e}")

# Parameters invullen
sender_email = "nnunet_training@outlook.com"
sender_password = "ErasmusmcRg6!"
recipient_email = "pb_vandenberg@outlook.com"
subject = "Test e-mail"
body = "Dit is een test-e-mail, verstuurd via Python."

# Functie aanroepen
send_email(sender_email, sender_password, recipient_email, subject, body)
