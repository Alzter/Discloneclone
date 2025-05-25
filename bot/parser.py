from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import glob

def parse_discord_conversation(c : pd.DataFrame) -> pd.DataFrame:
    def parse_time_str(time : str) -> datetime:
        return datetime.strptime(time,'%Y-%m-%dT%H:%M:%S.%f0%z')

    if type(c["Date"][0]) is str:
        c["Date"] = c["Date"].map(parse_time_str)
    c["Delay"] = c["Date"] - c["Date"].shift(1)
    c["Delay"] = c["Delay"].fillna( timedelta(seconds=0) )
    c = c.dropna(subset="Content").reset_index(drop=True)
    return c

def parse_discord_conversation_csv(path : str) -> pd.DataFrame:
    conversation = pd.read_csv(path)
    return parse_discord_conversation(conversation)

def get_chat(user : str, path : str) -> pd.DataFrame:
    path = path + "/Direct Messages - " + user + " [*.csv"
    conversation = glob.glob(path)

    if conversation:
        return parse_discord_conversation_csv(conversation[0])
    else:
        raise FileNotFoundError(f"No conversation(s) found at path {path}")

def split_by_conversations(messages : pd.DataFrame, gap_mins : 50, min_conv_length : int = 5) -> list[pd.DataFrame]:
    """
    Split a Discord conversation history into a list of shorter conversations separated by gap_mins minutes.

    Args:
        messages (DataFrame): Discord conversation history.
        gap_mins (int): How many minutes must have elapsed since the last message for the current message to be treated as the start of a new conversation.
        min_conv_length (int): If a conversation has less messages than this, don't include it in the list.
        max_conv_length (int): If a conversation has more messages than this, slice it up into chunks of this size.

    Returns:
        conversations (list[DataFrame]): List of all conversations ordered from least to most recent.
    """
    def is_new_conversation(delay : timedelta, max_delay_mins : int = gap_mins):
        """
        Given a delay between messages, asses whether the delay is sufficient enough
        for the message to be considered the start of a new conversation.
        """
        max_delay = timedelta(minutes = max_delay_mins)
        return delay > max_delay

    def get_conversation_indices(messages : pd.DataFrame) -> list[list[int]]:
        """
        Given a Discord conversation history with a boolean column "Start"
        denoting the start of a conversation, return a 2D list containing
        the indices of all messages grouped by conversation.

        Args:
            messages (DataFrame): Discord conversation history with "Start" column.
        Returns:
            conversation_indices (list[list[int]])
        """
        start_indices = messages[messages["Start"] == True].index

        indices = []
        for i in range(len(start_indices)):
            if i >= len(start_indices) - 1: continue
            indices.append(
                list(range(start_indices[i], start_indices[i+1]))
            )
        return indices

    messages["Start"] = messages["Delay"].map(is_new_conversation)
    messages.loc[0, "Start"] = True

    conversation_indices = get_conversation_indices(messages)

    conversations = []
    for indices in conversation_indices:

        if len(indices) < min_conv_length: continue
        conversation = messages.iloc[indices].reset_index(drop=True)
        conversations.append(conversation)

    return conversations

def get_header(messages : pd.DataFrame, target_user : str | None = None) -> str:
    """
    Return statistics about a Discord conversation (i.e., users involved, time)
    """
    end_time = messages.iloc[-1].Date

    users = messages.Author.unique()
    if target_user:
        if not target_user in users:
            users = np.insert(users, 0, target_user)

    string = "Conversation between " + ", ".join(users) + "."

    string += "\nObtained " + end_time.strftime("%y/%m/%d, %H:%M:%S") + "."

    return string

def to_string(messages : pd.DataFrame, context : str | None = None, header : bool = True, target_user : str | None = None) -> str:
    """
    Convert a Discord conversation history from DataFrame into a raw string.
    """
    string = ""

    if header:
        string += get_header(messages, target_user=target_user)

    if context: string += "\nContext of the conversation:\n" + context

    for i, message in messages.iterrows():
        string += f"\n\n{message.Author} {message.Date.strftime("%H:%M:%S")}"
        string += f"\n{message.Content}"

    return string.strip()
