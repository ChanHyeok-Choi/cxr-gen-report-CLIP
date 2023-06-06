import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import clip

class multi_label_loss(nn.Module):
    def __init__(self):
        super(multi_label_loss, self).__init__()

    def forward(self, predictions, target):
        predictions = torch.sigmoid(predictions)
        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, target)

        return torch.mean(loss)
    
class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

 
class Clip_loss(nn.Module):
    def __init__(self, lambda_direction=0., lambda_global=1., direction_loss_type='cosine', clip_model='ViT-B/32'):
        super(Clip_loss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model, self.clip_preprocess = clip.load(clip_model, device=self.device, jit=False)
        print("Loaded in pretrained model.")

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              self.clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              self.clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.model.load_state_dict(torch.load("/content/drive/MyDrive/UNIST/2023_1/NLP/ChestXrayReportGen/clip-imp-pretrained_128_6_after_4.pt", map_location=self.device))
        self.model = self.model.to(self.device)
        
        self.direction_loss = DirectionLoss(direction_loss_type)
        
        self.model.requires_grad_(False)
        
        self.lambda_global    = lambda_global
        self.lambda_direction = lambda_direction
        
        self.src_text_features = None
        self.target_text_features = None
        
        self.target_direction = None
        
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)
    
    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features
    
    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum() == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True))
        
        return self.direction_loss(edit_direction, self.target_direction).mean()
    
    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
            
        tokens = clip.tokenize(text).to(self.device)
        image  = self.preprocess(img)

        logits_per_image, _ = self.model(image.to(self.device), tokens)

        return (1. - logits_per_image / 100).mean()
    
    def forward(self, src_img: torch.Tensor, source_class, target_img: torch.Tensor, target_class, texture_image: torch.Tensor = None):
        clip_loss = 0.0

        # if self.lambda_global:
        #     clip_loss += self.lambda_global * self.global_clip_loss(target_img, [f"a {target_class}"])

        # if self.lambda_direction:
        #     clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img, target_class)

        # Image-Text direction similiarity loss
        # print(src_img.shape, target_img.shape)
        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        target_direction = (target_features - target_encoding)
        target_direction /= (target_direction.clone().norm(dim=-1, keepdim=True))
        source_direction = (source_features - src_encoding)
        source_direction /= (source_direction.clone().norm(dim=-1, keepdim=True))

        clip_loss = self.direction_loss(source_direction, target_direction).mean()

        return clip_loss
    
    
def test():
    target = torch.ones((13, 14))
    predictions = torch.randn((13, 14))
    print(target.shape, predictions.shape)
    loss_func = multi_label_loss()
    loss = loss_func(predictions, target)
    clip_loss_ = Clip_loss()
    image = torch.randn((224, 224, 3))
    text = ['dog']
    print(clip_loss_(image, text))
    print(loss)

if __name__=='__main__':
	test()