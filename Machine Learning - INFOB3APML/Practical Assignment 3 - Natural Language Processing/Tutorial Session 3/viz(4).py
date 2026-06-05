from typing import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from wordcloud import WordCloud
from matplotlib.patches import Rectangle


def show_topic_weights_and_counts(
        df_dominant_topics,
        col_topic_id="topic_id",
        col_topic="topic",
):
    """Shows topics and their weights as well as their counts

    Args:
        df_dominant_topics (pandas.DataFrame): DataFrame with coloumns containing following information: [sentence id, topic id, topic score]. The df needs to be ordered by topic score with the highest at the top. 
        col_* (str, optional): The column name for each important column if you deviate from the default names.
    """
    topk = 3
    topk_words_per_topic = {
        topic_id: "\n".join(eval(words)[:topk])
        for topic_id, words in list(df_dominant_topics.groupby([col_topic_id, col_topic]).groups)
    }
    weighted_topic_sums = df_dominant_topics.groupby(col_topic_id).topic_score.agg(
        ["sum", "count"]).rename(columns={
            "sum": "Sum of Topic Weights",
            "count": "Count of dominant topics"
        })
    df_top_topics = weighted_topic_sums.rename(index=topk_words_per_topic)
    df_top_topics.plot.barh(use_index=True, subplots=True, figsize=(20, 15), sharex=False, xlabel="Topic")

    plt.show()


def show_sentences(sentence_strongest_topic, per_sent_words_with_strongest_topic, start=0, end=13):
    """Shows the overall sentence topic and the individual words with their important topics 

    Args:
        sentence_strongest_topic (dict): Similar to per_sent_words_with_strongest_topic only in the form of {SENTENCE_ID:STRONGEST_TOPIC_ID}.
        per_sent_words_with_strongest_topic (dict): A nested dict in the form of {SENTENCE_ID:{WORD:STRONGEST_TOPIC_ID}}, where the types are int, string and int, respectively.
        start (int, optional): At which sentence to start. Defaults to 0.
        end (int, optional): Where to end. Defaults to 13.
    """
    mycolors = [color for _, color in mcolors.TABLEAU_COLORS.items()]
    sent_range = range(start, end)
    len_sentences = len(sent_range)
    fig, axes = plt.subplots(len_sentences, 1, figsize=(20, len_sentences * 0.95), dpi=160)
    axes[0].axis('off')
    for i, ax in zip(sent_range, axes):
        if i > 0:
            curr_idx = i - 1
            curr_sent = per_sent_words_with_strongest_topic[curr_idx]
            # curr_sent_words = list((word2id[word], val) for word, val in curr_sent.items())

            ax.text(0.01,
                    0.5,
                    "Doc " + str(i - 1) + ": ",
                    verticalalignment='center',
                    fontsize=16,
                    color='black',
                    transform=ax.transAxes,
                    fontweight=700)

            # Draw Rectange
            ax.add_patch(
                Rectangle((0.0, 0.05),
                          0.99,
                          0.90,
                          fill=None,
                          alpha=1,
                          color=mycolors[sentence_strongest_topic[curr_idx]],
                          linewidth=2))

            word_pos = 0.06
            if not len(curr_sent):
                ax.text(word_pos,
                        0.5,
                        f"Non of these words was assingned a topic: {[wd[0] for wd in curr_sent]}",
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16,
                        color="black",
                        transform=ax.transAxes,
                        fontweight=700)
                ax.axis('off')
                continue
            for j, (word, topics) in enumerate(curr_sent.items()):
                if j < 14:
                    ax.text(word_pos,
                            0.5,
                            word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16,
                            color=mycolors[topics],
                            transform=ax.transAxes,
                            fontweight=700)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    dots = '...'
                    ax.text(word_pos,
                            0.5,
                            dots,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16,
                            color='black',
                            transform=ax.transAxes)
                    word_pos += .004 * len(dots)
                ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end - 2),
                 fontsize=22,
                 y=0.95,
                 fontweight=700)
    plt.tight_layout()
    plt.show()


def show_wordcounts_and_topics(
        topics,
        words_in_flat_list,
        k=50,
        num_plts_per_row=3,
        col_topic_id="topic_id",
):
    """Shows the overall word counts as well as their weights within that topic

    Args:
        topics (dict): A nested dict in the form of {TOPIC_ID:{WORD:TOPIC_VALUE}}, where the types are int, string and float, respectively. Each topic should contain the at least the 10 most important terms for that topic.
        words_in_flat_list (list): All sentences in a flat list of words. E.g: [sent1_word1, sent1_word2, sent1_word3, sent2_word1, sent2_word3, ..., sentN_wordM]
        k (int, optional): The top k rows of df_dominat_topics to pic as a basis. Defaults to 50.
        num_plts_per_row (int, optional): [description]. Defaults to 3.
        col_* (str, optional): The column name for each important column if you deviate from the default names.
    """
    data_flat = words_in_flat_list
    counter = Counter(data_flat)
    out = []
    for i, topic in topics.items():
        for word, weight in topic.items():
            out.append([word, i, weight, counter[word]])

    df_word_dominant_topics = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    color_map = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    strongest_topics = sorted(df_word_dominant_topics.head(k).topic_id.unique())
    num_topics = len(strongest_topics)
    num_axes_to_delete = num_plts_per_row - (num_topics % num_plts_per_row)
    rows = (num_topics // num_plts_per_row) + 1
    fig, axes = plt.subplots(
        rows,
        num_plts_per_row,
        figsize=(20 * (num_topics // 3), 12),
        #  sharex=True,
        sharey=False)
    fig.set_size_inches((15, 3.5 * rows))

    for ax in axes[-1, -num_axes_to_delete:].flatten():
        fig.delaxes(ax)
        del ax

    for i, (ax, topic_id) in enumerate(zip(axes.flatten(), strongest_topics)):
        ax.bar(x='word',
               height="word_count",
               data=df_word_dominant_topics.loc[df_word_dominant_topics[col_topic_id] == i, :],
               color=color_map[i],
               width=0.5,
               alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word',
                    height="importance",
                    data=df_word_dominant_topics.loc[df_word_dominant_topics[col_topic_id] == i, :],
                    color=color_map[i],
                    width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=color_map[i])
        ax.set_title('Topic: ' + str(i), color=color_map[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df_word_dominant_topics.loc[df_word_dominant_topics[col_topic_id] == topic_id, 'word'],
                           rotation=30,
                           horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    plt.show()


def show_topic_wordclouds(
        df_dominant_topics,
        topics,
        k=50,
        num_plts_per_row=3,
        col_topic_id="topic_id",
):
    """This function shows the wordclouds of topics .

    Args:
        df_dominant_topics (pandas.DataFrame): DataFrame with coloumns containing following information: [topic id]. The df needs to be ordered by topic score with the highest at the top. 
        topics (dict): A nested dict in the form of {TOPIC_ID:{WORD:TOPIC_VALUE}}, where the types are int, string and float, respectively. Each topic should contain the at least the 10 most important terms for that topic.
        k (int, optional): The top k rows of df_dominat_topics to pic as a basis. Defaults to 50.
        num_plts_per_row (int, optional): Num of plots to be shown per row. Defaults to 3.
        col_* (str, optional): The column name for each important column if you deviate from the default names.
    """
    #
    # topics --
    color_map = [color for _, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    strongest_topics = sorted(df_dominant_topics.head(k)[col_topic_id].unique())
    num_topics = len(strongest_topics)
    num_axes_to_delete = num_plts_per_row - (num_topics % num_plts_per_row)
    rows = (num_topics // num_plts_per_row) + 1
    fig, axes = plt.subplots(rows, num_plts_per_row, sharex=True, sharey=False)
    fig.set_size_inches((15, 3.5 * rows))

    for ax in axes[-1, -num_axes_to_delete:].flatten():
        fig.delaxes(ax)

    for ax, topic_id in zip(axes.flatten(), strongest_topics):
        fig.add_subplot(ax)
        topic_words = topics[topic_id]
        cloud = WordCloud(background_color='white',
                          width=1000,
                          height=1000,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: color_map[topic_id],
                          prefer_horizontal=1.0)
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(topic_id), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


def show_topic_distributions(df_dominant_topics,
                             k=50,
                             num_plts_per_row=3,
                             col_topic_id="topic_id",
                             col_sent_len="sent_len"):
    """This function shows the distribution of topics.

    Args:
        df_dominant_topics (pandas.DataFrame): DataFrame with coloumns containing following information: [topic id, sentence len]. The df needs to be ordered by topic score with the highest at the top. 
        k (int, optional): The top k rows of df_dominat_topics to pic as a basis. Defaults to 50.
        num_plts_per_row (int, optional): Num of plots to be shown per row. Defaults to 3.
        col_* (str, optional): The column name for each important column if you deviate from the default names.
    """

    color_map = [color for _, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    strongest_topics = sorted(df_dominant_topics.head(k)[col_topic_id].unique())
    num_topics = len(strongest_topics)
    num_axes_to_delete = num_plts_per_row - (num_topics % num_plts_per_row)
    rows = (num_topics // num_plts_per_row) + 1
    fig, axes = plt.subplots(rows, num_plts_per_row, sharex=True, sharey=False)
    fig.set_size_inches((15, 3.5 * rows))
    for ax in axes[-1, -num_axes_to_delete:].flatten():
        fig.delaxes(ax)
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=20)

    for i, (ax, topic_id) in enumerate(zip(axes.flatten(), strongest_topics)):
        curr_topic_selection = df_dominant_topics[col_topic_id] == topic_id
        most_dominant_topic_sub = df_dominant_topics.loc[curr_topic_selection, :]
        sent_lens = most_dominant_topic_sub[col_sent_len]
        ax.hist(sent_lens, bins=20, color=color_map[i])
        ax.tick_params(axis='y', labelcolor=color_map[i], color=color_map[i])
        sns.kdeplot(sent_lens, color="black", shade=False, ax=ax.twinx())
        ax.set_xlabel(f'Sentence Word Count for Topic: {str(topic_id)}')
        ax.set_ylabel('Number of Documents', color=color_map[i])

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, right=1.1)
    plt.show()


def show_top_k_topics(df_dominant_topics,
                      top_k=5,
                      col_sent_id="sent_id",
                      col_topic_id="topic_id",
                      col_topic_score="topic_score"):
    """ This function shows the top5 and the least5 sentences and with their most dominant topic and the score for it.

    Args:
        df_dominant_topics (pandas.DataFrame): DataFrame with coloumns containing following information: [sentence id, topic id, topic score]. The df needs to be ordered by topic score with the highest at the top. 
        top_k (int, optional): How many bars will be shown. Defaults to 5.
        col_* (str, optional): The column name for each important column if you deviate from the default names. 
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(12, 10))

    ax = ax1
    subset_top = df_dominant_topics.head(top_k)
    subset_top.plot.bar(x=col_sent_id, y=col_topic_score, title="top5", use_index=True, ax=ax)
    rects = ax.patches
    labels = subset_top[col_topic_id]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height * .5, label, ha='center', va='bottom', color="white")

    ax = ax2
    subset_least = df_dominant_topics.tail(top_k)
    subset_least.plot.bar(x=col_sent_id, y=col_topic_score, title="least5", use_index=True, ax=ax)
    ax.xaxis.set_tick_params(rotation=90)
    rects = ax.patches
    labels = subset_least[col_topic_id]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height * .5, label, ha='center', va='bottom', color="white")

    fig.tight_layout()
    plt.show()