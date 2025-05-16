import re
import numpy as np

def sentence_split(text):
    single_sentences_list = re.split('([。；？！.;?!])', text)
    sentences = ["".join(i) for i in zip(single_sentences_list[0::2], single_sentences_list[1::2])]
    sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]
    return sentences


def combine_sentences(sentences, buffer_size=1):
    combined_sentences = [
        ' '.join(
            sentences[j]['sentence'] for j in range(max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences))))
        for i in range(len(sentences))
    ]
    for i, combined_sentence in enumerate(combined_sentences):
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity(embedding_current, embedding_next)
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance
    return distances, sentences


def chunk_com(distances, bpp_threshold=80):
    breakpoint_percentile_threshold = bpp_threshold
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    return indices_above_thresh, breakpoint_distance_threshold


def chunk_gen(indices_above_thresh, sentences):
    start_index = 0
    chunks = []
    chunks_len = []

    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        chunks_len.append(len(combined_text))

        start_index = index + 1

    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)
        chunks_len.append(len(combined_text))

    return chunks, chunks_len


def chunk_re_gen(chunks, chunks_len, max_chunk_len):
    chunks_new = []
    chunks_len_new = []
    chunk_len = 0
    chunk = ''
    for i, chunk_i in enumerate(chunks):
        chunk_len_new = chunk_len + chunks_len[i]
        chunk_new = chunk + chunk_i
        if chunk_len_new > max_chunk_len:
            chunks_new.append(chunk)
            chunks_len_new.append(chunk_len)
            chunk_len = chunks_len[i]
            chunk = chunk_i
        else:
            chunk_len = chunk_len_new
            chunk = chunk_new

    chunks_new.append(chunk)
    chunks_len_new.append(chunk_len)

    return chunks_new, chunks_len_new


def sentence_chunking(model, text, bpp_threshold):
    if type(text) == list:
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(text)]
    else:
        sentences = sentence_split(text)
    sentences = combine_sentences(sentences)
    embeddings = model.encode([x['combined_sentence'] for x in sentences])

    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]

    distances, sentences = calculate_cosine_distances(sentences)
    indices_above_thresh, breakpoint_distance_threshold = chunk_com(distances, bpp_threshold)
    chunks, chunks_len = chunk_gen(indices_above_thresh, sentences, )
    return chunks, chunks_len


def sentence_chunking_main(chunk_len, bpp_threshold, re_combine, text, model):
    max_chunk_len = chunk_len * 4

    chunks, chunks_len = sentence_chunking(model, text, bpp_threshold)

    while True:
        if len(chunks) > 1:
            chunks_new, chunks_len_new = sentence_chunking(model, chunks, bpp_threshold)
            if max(chunks_len_new) < max_chunk_len:
                chunks = chunks_new
                chunks_len = chunks_len_new
            else:
                break
        else:
            re_combine = False
            break


    if re_combine:
        chunks, chunks_len = chunk_re_gen(chunks, chunks_len, max_chunk_len)
    return chunks, chunks_len