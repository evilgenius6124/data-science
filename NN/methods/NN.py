'''CNN'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), #32 output channel
                 activation='relu',
                 input_shape=input_shape))   #input_shape = (img_x, img_y, 1) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))      #default strides is equal to pool_size
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
		
history = AccuracyHistory()  #history list train acc

filepath="weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
verbose=1, save_best_only=True, mode='max')

callbacks=[history, checkpoint]

************************************************************************

(batch_size, num_steps, hidden_size) #LSTM layer output

model = Sequential()
#Embedding(input_dim, output_dim, input_length)
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps)) 
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

*******************

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score > self.max_score):
                print("*** New High Score (previous: %.6f) \n" % self.max_score)
                model.save_weights("best_weights.h5")
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True
					
					
***************************

def get_model(num_filters, top_k):

    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, 2 * num_filters * top_k))

    inp = Input(shape=(maxlen, ))
    layer = Embedding(max_features, embed_size, weights=[embedding_matrix])
    x = layer(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(num_filters, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    k_max = Lambda(_top_k)(x)
    conc = concatenate([avg_pool, k_max])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
	
###################################################

    crawl-300d-2M.vec
    glove.840B.300d.w2vec.txt
    wiki.en.vec
    glove.twitter.27B.200d.txt
    GoogleNews-vectors-negative300.bin
    numberbatch-en.txt
    lexvec commoncrawl.
