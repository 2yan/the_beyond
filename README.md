# the_beyond
Data Processing Tools for Machine Learning and Data Analysis. Currently the only module is WordsWorth

## WordsWorth
WordsWorth is a text data preprocessing class that specializes in shaping, converting to one-hot vectors and examining the output of reccurant neural networks with the following properties: 
They need to takea 3 dimensional input of with the following shape: 
(observations,timesteps, unique_value_length) 
This library only processess text to one hot vectors, i.e the word 'hey' becomes: 
> array([[[0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]]])

It also does useful things like convert your output back into text. 

### QuickStart
>from the_beyond import WordsWorth  
sentence_length = 10 

>text = 'the sweet text to be trained on  '  
worth = WordsWorth(text)  
X, y = worth.get_sentences(letters)  
model.train(X, y)  

Then to generate seed data for your rnn, you would do the following: 

> seed_text = worth.generate_seed('seed text', letters)  
prediction = model.predict(seed_text)

Observing the results is as easy as: 
>worth.one_hot_to_text(prediction)



The Letters Argument basically just says how many letters you want to train your Rnn on. 
This how many letters your char-rnn will look at at a time, greater values indicate a longer short term memory.
Finally, 'model' over here refers to whatever neural network model you're using. 

#### Help I'm stuck ( and have Keras ):
If you're having a bunch of trouble with stuff not not matching up ( I know, I was confused by the extra dimensionality)

Then this example might help:

import keras as kr
>def generate_model(X, y):

    model = kr.models.Sequential()
    model.add(kr.layers.LSTM(128, input_shape=X.shape[1:]))
    model.add(kr.layers.Dense(X.shape[1]))
    model.add(kr.layers.Activation('softmax'))
    optimizer = kr.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model





