import torch
from models.resnet_rnn import ResNetRNN
import torch.nn.functional as F
import io
from ctc import  ctc_decode
from PIL import Image
from torchvision import transforms

class CaptchaSolver(object):
    """
    captcha sovler model to predict
    """
    def __init__(self,config:dict):
        super(CaptchaSolver, self).__init__()
        self.input_shape = config['input_shape']
        self.characters = config['characters']
        self.char2label = {char: i for i, char in enumerate(self.characters)}
        self.label2char = {label: char for char, label in self.char2label.items()}
        self.decode_method = config['decode_method']
        self.beam_size = config['beam_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = config['checkpoint']
        self.model = ResNetRNN(self.input_shape,len(self.characters),config['map_to_seq_hidden'],
                                                                                config['rnn_hidden'])
        self._load_model(self.checkpoint)


    def _transform_image(self,image_bytes):
        """

        :param image_bytes:
        :return:
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image = Image.open(io.BytesIO(image_bytes))
        print(image.size, image.mode)
        if image.mode == 'RGBA':
            r,g,b,a = image.split()
            image.load() # required for png.split()
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=a) # 3 is the alpha channel
            image  =  background
        w, h = image.size
        w = int(w*(self.input_shape[1]/h))
        img_transforms = transforms.Compose([transforms.Resize((self.input_shape[1],w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = mean,std = std)
                                            ])
        return img_transforms(image).unsqueeze(0)

    def _load_model(self, checkpoint):
        print('start load model')
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        print('load model finish')

    def predict(self, img_bytes):
        imgs = self._transform_image(img_bytes)
        print(imgs.shape)
        self.model.eval()
        with torch.no_grad():
            imgs.to(self.device)
            logits = self.model.forward(imgs)
            log_probs = F.log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, method=self.decode_method, beam_size=self.beam_size,
                               label2char=self.label2char)
            labels = []
            for pred in preds:
                labels.append(''.join(pred))
            return labels



