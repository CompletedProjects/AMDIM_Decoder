import os
import torch

from model import Model


class Checkpointer():
    def __init__(self, output_dir=None):
        # set output dir will this checkpoint will save itself
        self.output_dir = output_dir
        self.classifier_epoch = 0
        self.classifier_step = 0
        self.info_epoch = 0
        self.info_step = 0

        # Nawid - Added this information for the decoder
        self.decoder_epoch = 0
        self.decoder_step = 0

    def track_new_model(self, model):
        self.model = model

    def restore_model_from_checkpoint(self, cpt_path, training_classifier=False, training_decoder=False): # Nawid -  Restores a model from  a checkpoint
        ckp = torch.load(cpt_path)
        hp = ckp['hyperparams'] # Nawid - Hyperparameters from the checkpoint
        params = ckp['model'] # Nawwid - Parameters of the model
        self.info_epoch = ckp['cursor']['info_epoch']
        self.info_step = ckp['cursor']['info_step']
        self.classifier_epoch = ckp['cursor']['classifier_epoch'] # Nawid - Obtain classifier epoch
        self.classifier_step = ckp['cursor']['classifier_step'] # Nawid - Obtain classifier step

        # Nawid - Added this information for the decoder
        self.decoder_epoch = ckp['cursor']['decoder_epoch']
        self.decoder_step = ckp['cursor']['decoder_step']



        self.model = Model(ndf=hp['ndf'], n_classes=hp['n_classes'], n_rkhs=hp['n_rkhs'],
                           n_depth=hp['n_depth'], encoder_size=hp['encoder_size']) # Nawid -Instantiates the model with the specific parameters which are set from the hyperparameters from the loaded model
        skip_classifier = (training_classifier and self.classifier_step == 0)
        skip_decoder = (training_decoder and self.decoder_step == 0)
        if training_classifier and self.classifier_step == 0: # Nawid - loads encoder weights only so the classifier is trained from the beginning
            # If we are beginning the classifier training phase, we want to start
            # with a clean classifier
            model_dict = self.model.state_dict() # Nawid - state of the model
            partial_params = {k: v for k, v in params.items() if not k.startswith("evaluator.")} # Nawid - parameters realted to encoder only( not parameters of classifier)
            model_dict.update(partial_params)
            params = model_dict # Nawid - Sets the saved weights equal only the encoder values
        elif training_decoder and self.decoder_step ==0:            
            model_dict = self.model.state_dict() # Nawid - state of the model
            partial_params = {k: v for k, v in params.items() if not k.startswith("decoder.")} # Nawid - parameters related to encoder and classifier ( not related to decoder)
            params = model_dict # Nawid - Sets the saved weights equal only the encoder values
        self.model.load_state_dict(params) # Nawid - Loads the saved weights of the model


        print("***** CHECKPOINTING *****\n"
                "Model restored from checkpoint.\n"
                "Self-supervised training epoch {}\n"
                "Classifier training epoch {}\n"
                "Decoder training epoch {}\n" # Nawid - Added this
                "*************************"
                .format(self.info_epoch, self.classifier_epoch, self.decoder_epoch)) # Nawid - Added information related to decoder epoch
        return self.model

    def _get_state(self):
        return {
            'model': self.model.state_dict(),
            'hyperparams': self.model.hyperparams,
            'cursor': {
                'info_epoch': self.info_epoch,
                'info_step': self.info_step,
                'classifier_epoch': self.classifier_epoch,
                'classifier_step':self.classifier_step,
                'decoder_epoch': self.decoder_epoch, # Nawid - Added the decoder information to the state
                'decoder_step': self.decoder_step,
            }
        }

    def _save_cpt(self): # Nawid -  Saves a checkpoint
        f_name = 'amdim_cpt.pth'
        cpt_path = os.path.join(self.output_dir, f_name)
        # write updated checkpoint to the desired path
        torch.save(self._get_state(), cpt_path)
        return


    def update(self, epoch, step, classifier=False, decoder=False): # Nawid-updates the epoch and step when it saves
        if classifier:
            self.classifier_epoch = epoch
            self.classifier_step = step
        elif decoder:
            self.decoder_epoch = epoch
            self.decoder_step = step
        else:
            self.info_epoch = epoch
            self.info_step = step
        self._save_cpt()


    def get_current_position(self, classifier=False, decoder = False): # Nawid - Gets the current epoch and classifier
        if classifier:
            return self.classifier_epoch, self.classifier_step
        elif decoder:
            return self.decoder_epoch, self.decoder_step
        return self.info_epoch, self.info_step
