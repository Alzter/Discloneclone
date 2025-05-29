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

def to_string(messages : pd.DataFrame, header : bool = False, timestamp : bool = True, anonymize : bool = False, target_user : str | None = None, context : str | None = None) -> str:
    """
    Convert a Discord conversation history from DataFrame into a raw string.

    Args:
        messages (DataFrame): The conversation.
        header (bool, optional): Whether to include metadata at the start of the string, including the users in conversation and the time of the last message. Defaults to True.
        timestamp (bool, optional): Whether to include a timestamp for each message. Defaults to True.
        anonymize (bool, optional): Whether to replace all user names with "user", except for the target user. Defaults to False.
        target_user (str, optional): Which user to use as the focal point of the conversation. Defaults to None.
        context (str, optional): Additional context for the conversation. If provided, added after the header. Defaults to None.
    """
    string = ""

    if header:
        string += get_header(messages, target_user=target_user)

    if type(target_user) is str: target_user = target_user.lower()

    if context: string += "\nContext of the conversation:\n" + context

    for i, message in messages.iterrows():
        author = message.Author if (not anonymize or message.Author.lower() == target_user) else "user"
        string += "\n\n" + author
        if timestamp: string += " " + message.Date.strftime("%H:%M:%S")
        string += f"\n{message.Content}"

    return string.strip()

def to_dataset(messages : pd.DataFrame, target_user : str, window_size : int = 10, anonymise : bool = True, timestamp : bool = False):
    data = {"content" : [], "label" : []}

    # Use a sliding window to slice the conversation up
    slices = [msgs for msgs in messages.rolling(window_size)]

    # Get all messages from the target user
    target_user_indices = list(messages[messages.Author == target_user].index)
    other_user_indices = list(messages[messages.Author != target_user].index)
    
    # If there's no messages by the user and others, return nothing
    if not target_user_indices and not other_user_indices:
        return pd.DataFrame(data)

    # Remove all messages from the target user which don't come
    # before a different user's message at the start of the conversation
    target_user_indices = [i for i in target_user_indices if i > other_user_indices[0]]

    # Remove all slices which don't end with a message by the target user
    slices = [msgs for msgs in slices if msgs.index.stop - 1 in target_user_indices]

    
    for msgs in slices:
        inputs = msgs.iloc[:-1]
        output = msgs.iloc[-1]

        # Convert inputs/outputs to string
        inputs = to_string(inputs, header=False, timestamp=timestamp, anonymize=anonymise, target_user=target_user)
        output = output.Content

        data['content'].append(inputs)
        data['label'].append(output)

    return pd.DataFrame(data)

