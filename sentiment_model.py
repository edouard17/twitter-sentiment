from keras.models import Model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dim vectors.
    """
    input_dim = len(word_to_index) + 1
    output_dim = word_to_vec_map['hello'].shape[0]
    
    emb_matrix = np.zeros((input_dim, output_dim))
    for word, index in word_to_index.items():
        if word_to_vec_map[word].shape[0] == 50:
            emb_matrix[index, :] = word_to_vec_map[word]
        else:
            continue
    
    embedding_layer = Embedding(input_dim, output_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer



def sentiment_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function to create the model's graph.
    """
    text_indices = Input(input_shape, dtype='int32')   
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)   
    embeddings = embedding_layer(text_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation='sigmoid')(X)  
    model = Model(inputs=text_indices, outputs=X)
    return model

