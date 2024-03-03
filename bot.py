import telebot

from chat import Chat

import dotenv
dotenv.load_dotenv()
import os

bot = telebot.TeleBot(os.getenv("BOT_TOKEN"))

@bot.message_handler(commands=["start"], chat_types=['private'])
def start(message):
    bot.reply_to(message, """Use /ask to ask me for a movie.""")

@bot.message_handler(commands=["ask"], chat_types=['private'])
def ask(message):
    bot.reply_to(message, f"""Hey {message.from_user.first_name}
What movie do you want?""")
    chat.add_user(message.from_user.id)
    
@bot.message_handler(content_types=['text'], chat_types=['private'])
def get_response(message):
    if not chat.has_user(message.from_user.id):
        bot.reply_to(message, "Use /ask first.")
        return
    response = chat.chat(message.from_user.id, message.text)
    bot.reply_to(message, response)

@bot.message_handler(
    content_types=[
        'audio', 'document', 'animation', 'photo', 'sticker', 'video', 'voice'
    ],
    chat_types=['private'],
)
def unsupported_content(message):
    bot.reply_to(message, """Only text messages are supported.""")
    
@bot.message_handler(
    content_types=[
        'text', 'audio', 'document', 'animation', 'photo', 'sticker', 'video', 'voice'
    ],
    chat_types=['group', 'supergroup', 'channel'],
)
def unsupported_chat(message):
    bot.reply_to(message, """Only private chats are supported.""")

if __name__ == '__main__':
    chat = Chat()
    bot.polling()