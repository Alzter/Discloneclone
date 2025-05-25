import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import pandas as pd
import parser

load_dotenv()
token = os.getenv("DISCORD_TOKEN")

handler = logging.FileHandler(filename='discord.log',encoding='utf-8',mode='w')

intents = discord.Intents.default()

intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!',intents=intents)


@bot.event
async def on_ready():
    print("AIzter2 is running")

def to_dataframe(bot, messages : list):
    # Sort messages in chronological order
    if messages[-1].created_at < messages[0].created_at:
        messages.reverse()

    content = [message.clean_content for message in messages]
    #attachments = []
    author = [message.author.name for message in messages]
    authorid = [message.author.id for message in messages]
    date = [message.created_at for message in messages]
    
    # Swap the names of the real and AI alzter
    # so that the AI alzter assumes the position
    # of the real Alzter
    for i, name in enumerate(author):
        
        if name == "alzter":
            name = "AlzterBot"
        if name == bot.user.name:
            name = "alzter"

        author[i] = name

    data = {
        "AuthorID" : authorid,
        "Author" : author,
        "Date" : date,
        "Content" : content
    }

    data = pd.DataFrame(data)

    data = parser.parse_discord_conversation(data)

    return(data)

@bot.event
async def on_message(message):
    
    if message.author == bot.user: return
    history = [message async for message in message.channel.history(limit=10)]
     
    data = to_dataframe(bot, history)
    print(data)

    print(message.content)
    print(message.author)
    

    await message.channel.send("hi")
    #await message.channel.send(message.channel.id)

    await bot.process_commands(message)

bot.run(token, log_handler=handler, log_level=logging.DEBUG)
