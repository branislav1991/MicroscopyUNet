from keras.callbacks import Callback, ModelCheckpoint
import warnings
from sys import float_info
import math
from eval_mask_rcnn import eval_mAP
import os

class eval_checkpoint(Callback):
    def on_train_begin(self, logs={}):
        self.val_path='./data/stage1_val/'
        self.val_ids = next(os.walk(self.val_path))
        self.val_ids = [[self.val_ids[0] + d,d] for d in self.val_ids[1]]

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.model.config.AP_EVAL_FREQUENCY == 0:
            mAP = eval_mAP(self.val_ids, self.val_path, self.model)
            print(" — val_mAP: {0}".format(mAP))

class own_model_checkpoint(ModelCheckpoint):
    """This class is necessary to properly save our weight files.
    Keras does a good job saving weights if losses and metrics used are
    standard. For Mask RCNN, we need something more powerful.
    """
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(own_model_checkpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if "val_loss" in logs:
                logs = {"val_loss": logs["val_loss"][0][0]}
            else:
                logs = {"val_loss": math.nan}
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)