dependencies = ['torch']
from models import model_920 as model_920_
from models import model_921 as model_921_

# resnet18 is the name of entrypoint
def model_920():
    """ # This is model_920
    This function will return pretrained facenet model with accuracy 92%
    """
    # Call the model, load pretrained weights
    model = model_920_()
    return model

def model_921():
    """ # This is model_921
        This function will return pretrained facenet model with accuracy 92.135%
        """
    model = model_921_()