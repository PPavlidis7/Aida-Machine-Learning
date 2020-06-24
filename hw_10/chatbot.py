import glob
import json
import random
import os
import time
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from encoder_decoder import Encoder, BahdanauAttention, Decoder

NUM_DIALOGS = 55000


def read_data():
    dialogues_path = "./dialogues/*.txt"
    list_of_files = glob.glob(dialogues_path)
    list_of_dicts = []

    for file_name in list_of_files:
        with open(file_name) as f:
            for line in f:
                # each line is dictionary. We only need the 'turns' values
                list_of_dicts.append(json.loads(line)['turns'])
    return list_of_dicts


def preprocess_data(list_of_dicts):
    questions = []
    answers = []
    greetings = ['Hello', 'Hey', 'Hi']

    byes = ['Ok', 'Bye', 'Goodbye']

    for dictionary in list_of_dicts:
        dialogue = dictionary
        questions.append(random.choice(greetings))
        bot_flag = True

        for sentence in dialogue:
            if bot_flag:
                answers.append(sentence)  # Used for bot's answers
                bot_flag = False  # Switch
                continue
            else:
                questions.append(sentence)  # Used for user's questions
                bot_flag = True  # Switch

        if bot_flag:
            answers.append(random.choice(byes))

    answers, questions = shuffle(answers, questions)
    return answers, questions


def add_extra_tokens(matrix):
    # ref: https://www.tensorflow.org/tutorials/text/nmt_with_attention
    new_matrix = []
    for sequence in matrix:
        __sequence_copy = sequence
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Ref:
        # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        __sequence_copy = re.sub(r"([?.!,¿])", r" \1 ", __sequence_copy)
        __sequence_copy = re.sub(r'[" "]+', " ", __sequence_copy)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        __sequence_copy = re.sub(r"[^a-zA-Z?.!,¿]+", " ", __sequence_copy)

        __sequence_copy = __sequence_copy.rstrip().strip()

        sequence = "<start>" + " " + __sequence_copy + " " + "<end>"
        new_matrix.append(sequence)
    return new_matrix


def max_length(tensor):
    """ Returns the max length of a vector or a tensor """
    return max(len(t) for t in tensor)


def tokenize(text):
    """ Returns the padded text and the tokenizer, based on a text corpus """
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(text)
    tensor = tokenizer.texts_to_sequences(text)
    # Pad: https://www.tensorflow.org/guide/keras/masking_and_padding
    tensor = pad_sequences(tensor, padding='post')  # Pad to a fixed length
    return tensor, tokenizer


def generate_tensors_and_tokizers(input_text, target_text):
    """ Returns encoder and decoder padded texts & tokenizer, given the input and target text """
    input_tensor, inp_lang_tokenizer = tokenize(input_text)
    target_tensor, targ_lang_tokenizer = tokenize(target_text)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def init_checkpoint():
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    return checkpoint_prefix, checkpoint


def evaluate(sentence, units):
    # Preprocess the test sentence
    sentence = preprocess_sentence(sentence)

    sentence_array = sentence.split(' ')
    inputs = []

    for i in sentence_array:
        if i in inp_lang.word_index:
            inputs.append(inp_lang.word_index[i])

    # Pad, just like before
    inputs = pad_sequences([inputs], maxlen=max_length_input, padding='post')

    # Convert to tensor
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]  # Init hidden states to be passed to the encoder
    enc_out, enc_hidden = encoder(inputs, hidden)  # Encoder output & hidden states

    dec_hidden = enc_hidden  # Hidden states to the encoder are passed to the decoder

    # Decoder should know how to start
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    # Loop for maximum length of target
    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # Call argmax for the last Dense layer
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        # Stop when you predict the *end* token
        if targ_lang.index_word[predicted_id] == '<end>':
            result = result.replace('<end>', '')
            return result, sentence

        # The predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def preprocess_sentence(w):
    """ same preprocess with add_extra_tokens function """
    w = w.lower()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


def generate(sentence, units):
    result, sentence = evaluate(sentence, units)
    print('Input: %s' % (sentence))
    print('Predicted Bot Output: {}'.format(result))


@tf.function
def train_step(input, target, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(input, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(target[:, t], predictions)

            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def main():
    global BUFFER_SIZE, BATCH_SIZE, encoder, decoder, optimizer, targ_lang, inp_lang, max_length_target, \
        max_length_input

    list_of_data = read_data()
    answers, questions = preprocess_data(list_of_data)
    questions = questions[:NUM_DIALOGS]
    answers = answers[:NUM_DIALOGS]
    questions = add_extra_tokens(questions)
    answers = add_extra_tokens(answers)

    # Load texts as tensors & tokenizers for encoder/decoder
    input_tensor, target_tensor, inp_lang, targ_lang = generate_tensors_and_tokizers(questions, answers)

    # Calculate max_length of the target tensors
    max_length_target, max_length_input = max_length(target_tensor), max_length(input_tensor)

    # Creating training and validation sets using an 80%-20% split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 8
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = 50
    units = 200
    vocab_input_size = len(inp_lang.word_index) + 1
    vocab_target_size = len(targ_lang.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Instantiate the encoder with the corresponding vocabulary input size, embedding dimensions, units and batch size
    encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    checkpoint_prefix, checkpoint = init_checkpoint()

    EPOCHS = 5

    for epoch in range(EPOCHS):

        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch,
                                                             batch_loss.numpy()))

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for one epoch {} sec\n'.format(time.time() - start))

    print(encoder.summary())
    print("Enter -q- to quit")
    while True:
        user_input = str(input("User: "))
        if user_input == '-q-':
            print("Quitting chat..")
            break
        else:
            generate(user_input, units)


if __name__ == '__main__':
    print('-' * 10, '\n' * 3)
    main()
