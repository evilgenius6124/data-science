'''RNN model'''
def get_simple_rnn_model():
    embedding_dim = 300
    embedding_matrix = np.random.random((MAX_NB_WORDS, embedding_dim))
    
    inp = Input(shape=(MAX_LENGTH, ))
    x = Embedding(input_dim=MAX_NB_WORDS, 
	output_dim=embedding_dim, input_length=MAX_LENGTH, 
                  weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
	
rnn_simple_model = get_simple_rnn_model()


'''print layer'''
plot_model(rnn_simple_model, 
           to_file='./images/article_5/rnn_simple_model.png', 
           show_shapes=True, 
           show_layer_names=True

'''autometically save best model'''

filepath="./models/rnn_no_embeddings/weights-
improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 2

history = rnn_simple_model.fit(x=padded_train_sequences, 
                    y=y_train, 
                    validation_data=(padded_test_sequences, y_test), 
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)

best_rnn_simple_model = load_model('./models/rnn_no_embeddings/
weights-improvement-01-0.8262.hdf5')

y_pred_rnn_simple = best_rnn_simple_model.predict(padded_test_sequences, 
verbose=1, batch_size=2048)

##############################################



