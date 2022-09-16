import enum
from sqlite3 import paramstyle
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from end2you.utils import Params

from model import WAV2VEC2_BASE_PATH, get_feature_dim
import dataset

"""
Model definitions
"""

def count_all_parameters(model:torch.nn.Module) -> int:

    return sum([p.numel() for p in model.parameters()])


def count_trainable_parameters(model:torch.nn.Module) -> int:

    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def load_ssl_model(params:Params) -> nn.Module:
    """
    Loads a SSL Transformer model
    :params model Params object
    """

    ssl_model = params.feature_extractor
    #SpecAugment Args
    mask_time_prob = params.augment.mask_time_prob
    mask_time_length = params.augment.mask_time_length
    mask_feature_prob = params.augment.mask_feature_prob
    mask_feature_length = params.augment.mask_feature_length

    if "wav2vec2-base" in ssl_model:
        #model = transformers.Wav2Vec2Model.from_pretrained(WAV2VEC2_BASE_PATH,
        model = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base",
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
        mask_feature_prob=mask_feature_prob,
        mask_feature_length=mask_feature_length)
    else:
        raise NotImplementedError

    return model

class BaseMultiModule(nn.Module):
    """
    Baseline Model which routes the features from the extractor through independent shallow networks
    """
    def __init__(self, feat_dim:int, params:Params) -> None:
        super().__init__()
        self.params = params
        self.is_training = False

        # attention pooling of the features for each task
        if self.params.pool == "attention":
            self.pools = nn.ModuleList([nn.Linear(feat_dim, 1, bias=False) for i in range(params.num_outputs)])
        else: # avg pool 
            self.pools = []

        embedding_size = params.embedding_size

        # output heads - 2 layer networks
        self.voc_type_model = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 8),
            nn.BatchNorm1d(8),
        )
        self.low_model = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 2),
            nn.BatchNorm1d(2),
            #nn.Sigmoid(),   # no sigmoid for valence/arousal
        )
        self.high_model = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 10),
            nn.BatchNorm1d(10),
            nn.Sigmoid()
        )
        self.culture_emotion_model = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 40),
            nn.BatchNorm1d(40),
            nn.Sigmoid(),
        )

    def forward(self, inputs, batch=None):
        """
        inputs: [B, seqlen, featdim]
        """
        # first pool the features over time
        if self.params.pool == "attention":
            # attention per output
            voc_type_feat = torch.sum(torch.softmax(self.pools[0](inputs), dim=1) * inputs, dim=1)
            low_feat = torch.sum(torch.softmax(self.pools[1](inputs), dim=1) * inputs, dim=1)
            high_feat = torch.sum(torch.softmax(self.pools[2](inputs), dim=1) * inputs, dim=1)
            emotion_culture_feat = torch.sum(torch.softmax(self.pools[3](inputs), dim=1) * inputs, dim=1)

            # pass
            out_voc = self.voc_type_model(voc_type_feat)
            out_low = self.low_model(low_feat)
            out_high = self.high_model(high_feat)
            out_culture_emotion = self.culture_emotion_model(emotion_culture_feat)     


        else: # avg pool
            inputs = torch.mean(inputs, dim=1)
            # each head gets the same averaged features
            out_voc = self.voc_type_model(inputs)
            out_low = self.low_model(inputs)
            out_high = self.high_model(inputs)
            out_culture_emotion = self.culture_emotion_model(inputs)

        # return a dict per task
        return {"voc_type": out_voc, "low": out_low, "high": out_high, "culture_emotion": out_culture_emotion}


class StackedModule(nn.Module):
    """
    Stack model which sends the inputs through multiple heads, concatenating features with the output from the lower stages.
    It goes in a fixed order type -> low -> high -> culture
    """

    def __init__(self, feat_dim:int, params:Params) -> None:
        super().__init__()
        self.params = params

        # is training flag to switch chain from GT to predictions 
        # self.is_training = False  not needed since already in nn.Module

        # attention pooling of the features for each task
        if self.params.pool == "attention":
            #self.pools = nn.ModuleList([nn.Linear(feat_dim, 1, bias=False) for i in range(params.num_tasks)])
            self.pools = nn.Linear(feat_dim, 1, bias=False)
        else: # avg pool 
            #self.pools = []
            self.pools = None

        embedding_size = params.embedding_size

        # encoders
        self.voc_type_encoder = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 8),
            nn.BatchNorm1d(8),
        )
        # feeds into
        low_feat_dim =  feat_dim + 8
        self.low_encoder = nn.Sequential(
            #nn.BatchNorm1d(low_feat_dim),
            nn.Linear(low_feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 2), 
            nn.BatchNorm1d(2),
            #nn.Sigmoid()   # no sigmoid for valence/arousal
        )
        # feeds into
        high_feat_dim = low_feat_dim + 2
        self.high_encoder = nn.Sequential(
            nn.Linear(high_feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 10),
            nn.BatchNorm1d(10),
            nn.Sigmoid()
        )
        # feeds into
        culture_emotion_dim = high_feat_dim + 10
        self.culture_emotion_encoder = nn.Sequential(
            nn.Linear(culture_emotion_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 40),
            nn.BatchNorm1d(40),
            nn.Sigmoid()
        )

    def forward(self, inputs:torch.Tensor, batch:dict):
        """
        inputs: [B, seqlen, feat_dim]
        """

        # first aggregate the features over time
        if self.params.pool == "attention":
            weights = torch.softmax(self.pools(inputs), dim=1)
            inputs = torch.sum(weights * inputs, dim=1)
        else:   # avg pool
            inputs = torch.mean(inputs, 1)
        
        # during training time, the ground truth is fed to the model stages. When evaluating, the encoder outputs are used.

        voc_type_label = batch.get("voc_type")  # [B, 8]
        low_label = batch.get("low")    # [B, 2]
        high_label = batch.get("high")  # [B, 10]
        culture_emotion_label = batch.get("culture_emotion")    # [B, 40]

        #input_feat = inputs # update this variable across the chain

        # Type
        voc_type_pred = self.voc_type_encoder(inputs)
        # check if GT exists or we are in prediction mode
        if voc_type_label is None or not self.training:  # test or val case
            # add a softmax to the type predictions 
            vl = torch.softmax(voc_type_pred, dim=-1)
            low_input_feat = torch.cat([inputs, vl], dim=-1)
        elif voc_type_label is not None and self.training:   # train case
            # convert the voc_type to one hot encoding
            vl = F.one_hot(voc_type_label, 8)
            low_input_feat = torch.cat([inputs, vl], dim=-1).type_as(inputs)
        else:   # training with no GT label
            raise NotImplementedError

        # Low
        low_pred = self.low_encoder(low_input_feat)
        # check if GT exists or we are in prediction mode
        if low_label is None or not self.training:
            high_input_feat = torch.cat([low_input_feat, low_pred], dim=-1)
        elif low_label is not None and self.training:    # train case
            high_input_feat = torch.cat([low_input_feat, low_label], dim=-1)

        # High
        high_pred = self.high_encoder(high_input_feat)
        # check if GT exists or we are in prediction mode
        if high_label is None or not self.training:
            culture_emotion_input_feat = torch.cat([high_input_feat, high_pred], dim=-1)
        elif high_label is not None and self.training:
            culture_emotion_input_feat = torch.cat([high_input_feat, high_label], dim=-1)

        # culture specific emotion
        culture_emotion_pred = self.culture_emotion_encoder(culture_emotion_input_feat)

        return {"voc_type": voc_type_pred, "low": low_pred, "high": high_pred, "culture_emotion": culture_emotion_pred}


class StackedModuleV2(nn.Module):
    """
    new iteration of the classifier chain. Will permute the emotions to the order of performance in the baseline (descending), 
    then revert the permutation at the output to maintain compatibility with the rest of the code
    It goes in the order low->high->culture->type 
    """

    def __init__(self, feat_dim:int, params:Params) -> None:
        super().__init__()
        self.params = params

        # attention pooling of the features for each task
        if self.params.pool == "attention":
            self.pools = nn.Linear(feat_dim, 1, bias=False)
        else: # avg pool 
            self.pools = None

        embedding_size = params.embedding_size

        # output chain classifiers

        self.chain_high = False
        self.chain_culture = True

        self.high_culture_parallel = False

        if self.chain_high:
            print("Classifier chain for emotion")

        if self.chain_culture:
            print("Classifier chaon for culture")

        # input dimensions for the first layer of each task network
        voc_type_dim = feat_dim
        low_dim = voc_type_dim + 8
        high_dim = low_dim + 2
        
        if self.high_culture_parallel:
            print("High and Culture tasks run parallel to reduce chain length")
            culture_emotion_dim = high_dim # parallel
        else:
            culture_emotion_dim = high_dim + 10 # sequential

        
        #low_dim = feat_dim
        #high_dim = low_dim + 2
        #culture_dim = high_dim + 10
        #voc_type_dim = culture_dim + 40


        # low chains valence -> arousal
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU()
        )   # this part is shared for the low chain
        self.low_regressor = nn.ModuleList([
            nn.Sequential(
                #nn.Linear(low_dim + i, embedding_size),
                #nn.BatchNorm1d(embedding_size),
                #nn.GELU(),
                nn.Linear(embedding_size + i, 1),
                nn.BatchNorm1d(1),
                # no activation for VA
            ) for i in range(2)
        ])  # list of shallow networks with increasing input width

        if self.chain_high:
            # high chains 10 emotions in desc order of perf
            self.high_encoder = nn.Sequential(
                nn.Linear(high_dim, embedding_size),
                nn.BatchNorm1d(embedding_size),
                nn.GELU()
            )    # shared encoder network 
            self.high_regressor = nn.ModuleList([
                nn.Sequential(
                    #nn.Linear(high_dim + i, embedding_size),
                    #nn.BatchNorm1d(embedding_size),
                    #nn.GELU(),
                    nn.Linear(embedding_size + i, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid()
                ) for i in range(10)
            ]) # list of shallow networks with increasing input width

        else:
            self.high_encoder = nn.Sequential(
                nn.Linear(high_dim, embedding_size),
                nn.BatchNorm1d(embedding_size),
                nn.GELU(),
                nn.Linear(embedding_size, 10),
                nn.BatchNorm1d(10),
                nn.Sigmoid()
            )
        

        if self.chain_culture:
            # culture chains 4x10 emotions in desc order of perf per culture
            
            self.culture_emotion_encoder = nn.Sequential(
                nn.Linear(culture_emotion_dim, embedding_size),
                nn.BatchNorm1d(embedding_size),
                nn.GELU()
            )   # shared encoder network

            # 4 separate chains per culture
            self.china_regressor = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + i, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid(),
                ) for i in range(10)
            ])

            self.united_states_regressor = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + i, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid(),
                ) for i in range(10)
            ])

            self.south_africa_regressor = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + i, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid(),
                ) for i in range(10)
            ])

            self.venezuela_regressor = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_size + i, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid(),
                ) for i in range(10)
            ])
            """
            self.culture_emotion_regressor = nn.ModuleList([
                nn.Sequential(
                    #nn.Linear(culture_dim + i, embedding_size),
                    #nn.BatchNorm1d(embedding_size),
                    #nn.GELU(),
                    nn.Linear(embedding_size + i, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid(),
                ) for i in range(40)
            ]) 
            # list of shallow networks with increasing input width
            """
            # test - culture not chaining
        else:
            self.culture_emotion_encoder = nn.Sequential(
                nn.Linear(culture_emotion_dim, embedding_size),
                nn.BatchNorm1d(embedding_size),
                nn.GELU(),
                nn.Linear(embedding_size, 40),
                nn.BatchNorm1d(40),
                nn.Sigmoid()
            )
        


        # vocal type does not chain
        self.voc_type_encoder = nn.Sequential(
            nn.Linear(voc_type_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 8),
            nn.BatchNorm1d(8),
        )

    def forward(self, inputs:torch.Tensor, batch:dict):
        """
        inputs: [B, seqlen, feat_dim]
        """

        # first aggregate the features over time
        if self.params.pool == "attention":
            weights = torch.softmax(self.pools(inputs), dim=1)
            inputs = torch.sum(weights * inputs, dim=1)
        else:   # avg pool
            inputs = torch.mean(inputs, 1)
        
        # during training time, the ground truth is fed to the model stages. When evaluating, the encoder outputs are used.

        # these are in standard column order
        voc_type_label = batch.get("voc_type")  # [B, 8]
        low_label = batch.get("low")    # [B, 2]
        high_label = batch.get("high")  # [B, 10]
        culture_emotion_label = batch.get("culture_emotion")    # [B, 40]

        # permute the labels
        if high_label is not None:
            high_label_permuted = high_label[:, dataset.EMOTIONS_INDICES_PERMUTED]
            assert high_label_permuted.size() == high_label.size()
        else:
            high_label_permuted = None
        
        if culture_emotion_label is not None:
            china_emotion_label = culture_emotion_label[:,:10]
            us_emotion_label = culture_emotion_label[:,10:20]
            south_africa_emotion_label = culture_emotion_label[:,20:30]
            venezuela_emotion_label = culture_emotion_label[:,30:]

            china_emotion_label_permuted = china_emotion_label[:, dataset.CN_INDICES_PERMUTED]
            us_emotion_label_permuted = us_emotion_label[:, dataset.US_INDICES_PERMUTED]
            south_africa_emotion_label_permuted = south_africa_emotion_label[:, dataset.SA_INDICES_PERMUTED]
            venezuela_emotion_label_permuted = venezuela_emotion_label[:, dataset.VZ_INDICES_PERMUTED]

        #    culture_emotion_label_permuted = culture_emotion_label[:, dataset.CULTURE_EMOTIONS_INDICES_PERMUTED]
        #    assert culture_emotion_label_permuted.size() == culture_emotion_label.size()
        else:
            china_emotion_label = None
            us_emotion_label = None
            south_africa_emotion_label = None
            venezuela_emotion_label = None

            china_emotion_label_permuted = None
            us_emotion_label_permuted = None
            south_africa_emotion_label_permuted = None
            venezuela_emotion_label_permuted = None
            #culture_emotion_label_permuted = None


        # vocal type prediction
        voc_type_pred = self.voc_type_encoder(inputs)

        if voc_type_label is None or not self.training:  # test or val case
            # add a softmax to the type predictions 
            vl = torch.softmax(voc_type_pred, dim=-1)
            low_input_feat = torch.cat([inputs, vl], dim=-1)
        elif voc_type_label is not None and self.training:   # train case
            # convert the voc_type to one hot encoding
            vl = F.one_hot(voc_type_label, 8)
            low_input_feat = torch.cat([inputs, vl], dim=-1).type_as(inputs)
        else:   # training with no GT label
            raise NotImplementedError

        # low prediction
        low_pred = []
        #low_input_feat = inputs # start with just the inputs (features)

        high_input_feat = low_input_feat

        low_embedding = self.low_encoder(low_input_feat)    # [B, d_emb]
        for i in range(len(self.low_regressor)):
            lp = self.low_regressor[i](low_embedding)  # call in order
            low_pred.append(lp)
            if low_label is None or not self.training: # concat the pred
                low_embedding = torch.cat([low_embedding, lp], dim=-1)
                high_input_feat = torch.cat([high_input_feat, lp], dim=-1)
            elif low_label is not None and self.training:
                gt = torch.unsqueeze(low_label[:, i], dim=-1) # training case, append the permuted label
                low_embedding = torch.cat([low_embedding, gt], dim=-1)   # order is hardcoded for valence, arousal. different for emos
                high_input_feat = torch.cat([high_input_feat, gt], dim=-1)
        # combine preds into one tensor
        low_pred = torch.cat(low_pred, dim=-1)

        # high prediction
        if self.chain_high:
               
            high_pred = []

            culture_emotion_input_feat = high_input_feat

            high_embedding = self.high_encoder(high_input_feat)

            for i in range(len(self.high_regressor)):
                hp = self.high_regressor[i](high_embedding)   #call in order
                high_pred.append(hp)
                if high_label is None or not self.training:  
                    high_embedding = torch.cat([high_embedding, lp], dim=-1) #append the pred
                    if not self.high_culture_parallel:
                        culture_emotion_input_feat = torch.cat([culture_emotion_input_feat, lp], dim=-1)
                elif high_label is not None and self.training:  # training case, append the permuted label
                    if self.params.chain_order == "perf":
                        gt = torch.unsqueeze(high_label_permuted[:, i], dim=-1) 
                    else:
                        gt = torch.unsqueeze(high_label[:, i], dim=-1)
                    high_embedding = torch.cat([high_embedding, gt], dim=-1)
                    if not self.high_culture_parallel:
                        culture_emotion_input_feat = torch.cat([culture_emotion_input_feat, gt], dim=-1)
            # combine preds into one tensor
            high_pred = torch.cat(high_pred, dim=-1)
        
        else:    
            high_pred = self.high_encoder(high_input_feat)

            if not self.high_culture_parallel:  # use high task as input for culture task
                if high_label is None or not self.training:
                    culture_emotion_input_feat = torch.cat([high_input_feat, high_pred], dim=-1)
                elif high_label is not None and self.training:
                    culture_emotion_input_feat = torch.cat([high_input_feat, high_label], dim=-1)
        

        # culture prediction

        
        #culture_emotion_pred = []

        #vocal_input_feat = culture_input_feat

        if self.chain_culture:

            # china
            china_embedding = self.culture_emotion_encoder(culture_emotion_input_feat)

            china_pred = []

            for i in range(len(self.china_regressor)):
                us_p = self.china_regressor[i](china_embedding)
                china_pred.append(us_p)
                if china_emotion_label is None or not self.training:
                    china_embedding = torch.cat([china_embedding, us_p], dim=-1)
                elif china_emotion_label is not None and self.training:
                    if self.params.chain_order == "perf":
                        gt = torch.unsqueeze(china_emotion_label_permuted[:, i], dim=-1)
                    else:
                        gt = torch.unsqueeze(china_emotion_label[:, i], dim=-1)
                    china_embedding = torch.cat([china_embedding, gt], dim=-1)
            # combine preds into one tensor
            china_pred = torch.cat(china_pred, dim=-1)

            # us
            united_states_embedding = self.culture_emotion_encoder(culture_emotion_input_feat)

            united_states_pred = []

            for i in range(len(self.united_states_regressor)):
                us_p = self.united_states_regressor[i](united_states_embedding)
                united_states_pred.append(us_p)
                if us_emotion_label is None or not self.training:
                    united_states_embedding = torch.cat([united_states_embedding, us_p], dim=-1)
                elif us_emotion_label is not None and self.training:
                    if self.params.chain_order == "perf":
                        gt = torch.unsqueeze(us_emotion_label_permuted[:, i], dim=-1)
                    else:
                        gt = torch.unsqueeze(us_emotion_label[:, i], dim=-1)
                    united_states_embedding = torch.cat([united_states_embedding, gt], dim=-1)
            # combine preds into one tensor
            united_states_pred = torch.cat(united_states_pred, dim=-1)


            # south africa

            south_africa_embedding = self.culture_emotion_encoder(culture_emotion_input_feat)

            south_africa_pred = []

            for i in range(len(self.south_africa_regressor)):
                sa_p = self.south_africa_regressor[i](south_africa_embedding)
                south_africa_pred.append(sa_p)
                if south_africa_emotion_label is None or not self.training:
                    south_africa_embedding = torch.cat([south_africa_embedding, sa_p], dim=-1)
                elif south_africa_emotion_label is not None and self.training:
                    if self.params.chain_order == "perf":
                        gt = torch.unsqueeze(south_africa_emotion_label_permuted[:, i], dim=-1)
                    else:
                        gt = torch.unsqueeze(south_africa_emotion_label[:, i], dim=-1)
                    south_africa_embedding = torch.cat([south_africa_embedding, gt], dim=-1)
            # combine preds into one tensor
            south_africa_pred = torch.cat(south_africa_pred, dim=-1)

            # venezuela

            venezuela_embedding = self.culture_emotion_encoder(culture_emotion_input_feat)

            venezuela_pred = []

            for i in range(len(self.venezuela_regressor)):
                vz_p = self.venezuela_regressor[i](venezuela_embedding)
                venezuela_pred.append(vz_p)
                if venezuela_emotion_label is None or not self.training:
                    venezuela_embedding = torch.cat([venezuela_embedding, vz_p], dim=-1)
                elif venezuela_emotion_label is not None and self.training:
                    if self.params.chain_order == "perf":
                        gt = torch.unsqueeze(venezuela_emotion_label_permuted[:, i], dim=-1)
                    else:
                        gt = torch.unsqueeze(venezuela_emotion_label[:, i], dim=-1)
                    venezuela_embedding = torch.cat([venezuela_embedding, gt], dim=-1)
            # combine preds into one tensor
            venezuela_pred = torch.cat(venezuela_pred, dim=-1)
        
        else:
            culture_emotion_pred = self.culture_emotion_encoder(culture_emotion_input_feat)


        """
        for i in range(len(self.culture_emotion_regressor)):
            cp = self.culture_emotion_regressor[i](culture_embedding)
            culture_emotion_pred.append(cp)
            if culture_emotion_label is None or not self.training:
                culture_embedding = torch.cat([culture_embedding, cp], dim=-1)    # append the pred
                #vocal_input_feat = torch.cat([vocal_input_feat, lp], dim=-1)
            elif culture_emotion_label_permuted is not None and self.training: # training case, append the permuted label
                if self.params.chain_order == "perf":
                    gt = torch.unsqueeze(culture_emotion_label_permuted[:, i], dim=-1)
                else:
                    gt = torch.unsqueeze(culture_emotion_label[:, i], dim=-1)
                culture_embedding = torch.cat([culture_embedding, gt], dim=-1)
                #vocal_input_feat = torch.cat([vocal_input_feat, gt], dim=-1)
        # combine preds into one tensor
        culture_emotion_pred = torch.cat(culture_emotion_pred, dim=-1)

        """

        # vocal type prediction
        #voc_type_pred = self.voc_type_encoder(vocal_input_feat)
        if self.chain_high:
            if self.params.chain_order == "perf":  # the high and culture encoders are trained with the permuted order of labels. 
            
            # So the prediction tensors must be un-permuted here to restore the original order.
                high_pred = high_pred[:, dataset.EMO_INDICES_RESTORED]
            #culture_emotion_pred = culture_emotion_pred[:, dataset.CULTURE_EMOTIONS_INDICES_RESTORED]

        if self.chain_culture:
            if self.params.chain_order == "perf":  # the high and culture encoders are trained with the permuted order of labels. 
                china_pred = china_pred[:, dataset.CN_INDICES_RESTORED]
                united_states_pred = united_states_pred[:, dataset.US_INDICES_RESTORED]
                south_africa_pred = south_africa_pred[:, dataset.SA_INDICES_RESTORED]
                venezuela_pred = venezuela_pred[:, dataset.VZ_INDICES_RESTORED]

            culture_emotion_pred = torch.cat([china_pred, united_states_pred, south_africa_pred, venezuela_pred], dim=-1)


        return {"voc_type": voc_type_pred, "low": low_pred, "high": high_pred, "culture_emotion": culture_emotion_pred}



class AttentionStackModule(nn.Module):
    """
    Following Triantafyllopoulos et al. (2022), use multiple encoders and attention between the features. Stack these like the chain in Xin et al. (2022)
    """

    def __init__(self, feat_dim:int, params:Params) -> None:
        super().__init__()
        self.params = params

        embedding_size = params.embedding_size

        # common feature extractor or multiple encoders? 
        
        # mappings
        self.mappings = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(feat_dim, embedding_size),
                nn.GELU(),
                nn.Dropout(0.2),
            ) for _ in range(4)]
        )

        # mhas
        self.mhas = nn.ModuleList(
            modules=[nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, dropout=0.3, batch_first=True) for _ in range(3)]
        )

        # outputs

        # output networks
        self.voc_type_classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 8),
            nn.BatchNorm1d(8)

        )
        self.low_regressor = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 2),
            nn.BatchNorm1d(2)

        )
        self.high_regressor = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 10),
            nn.BatchNorm1d(10),
            nn.Sigmoid()

        )
        self.culture_regressor = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 40),
            nn.BatchNorm1d(40),
            nn.Sigmoid()

        )

    def forward(self, inputs, batch=None):
        """
        call the modules, pass on embeddings
        :inputs Tuple or list of 4 features returned by the 4 encoders. Expected to have size [B, L, feat_dim]
        """

        assert isinstance(inputs, list) or isinstance(inputs, tuple)

        # reduce dimensionality

        voc_type_features = self.mappings[0](inputs[0])
        low_features = self.mappings[1](inputs[1])
        high_features = self.mappings[2](inputs[2])
        culture_emotion_features = self.mappings[3](inputs[3])

        # apply mha sequentially

        low_features = self.mhas[0](query=low_features, key=voc_type_features, value=voc_type_features)[0]
        high_features = self.mhas[1](query=high_features, key=low_features, value=low_features)[0]
        culture_emotion_features = self.mhas[2](query=culture_emotion_features, key=high_features, value=high_features)[0]

        # temporal pooling
        if self.params.pool == "avg":
            voc_type_features = torch.mean(voc_type_features, dim=1)
            low_features = torch.mean(low_features, dim=1)
            high_features = torch.mean(high_features, dim=1)
            culture_emotion_features = torch.mean(culture_emotion_features, dim=1)
        else:   # pick last time step
            voc_type_features = voc_type_features[:, -1, :]
            low_features = low_features[:, -1, :]
            high_features = high_features[:, -1, :]
            culture_emotion_features = culture_emotion_features[:, -1, :]

        # calc outputs
        voc_type_pred = self.voc_type_classifier(voc_type_features)
        low_pred = self.low_regressor(low_features)
        high_pred = self.high_regressor(high_features)
        culture_emotion_pred = self.culture_regressor(culture_emotion_features)

        return {"voc_type": voc_type_pred, "low": low_pred, "high": high_pred, "culture_emotion": culture_emotion_pred}        



class AttentionBranchModule(nn.Module):
    """
    Branches out attention heads from the intermediate layers of the encoder network. Heads are connected in series, creating a parallel path to the backbone.
    Inspired by Song et al. (2022)
    """

    def __init__(self, params:Params) -> None:
        super().__init__()
        self.params = params

        conv_dim = 512
        tf_dim = 768
        embedding_size = params.embedding_size

        # size reductions
        self.conv_mapping = nn.Sequential(
            nn.Linear(conv_dim, embedding_size),
            #nn.BatchNorm1d(embedding_size),
            nn.GELU()
        )
        self.transformer_mappings = nn.ModuleList(
            modules=[nn.Sequential(
                nn.Linear(tf_dim, embedding_size),
                #nn.BatchNorm1d(embedding_size),
                nn.GELU()
                )
            for _ in range(4)] 
        )
        # multi head attention blocks
        self.mhas = nn.ModuleList(
            modules=[nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, dropout=0.3, batch_first=True) for _ in range(4)]
        )
        # output networks
        self.voc_type_classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 8),
            nn.BatchNorm1d(8)

        )
        self.low_regressor = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 2),
            nn.BatchNorm1d(2)

        )
        self.high_regressor = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 10),
            nn.BatchNorm1d(10),
            nn.Sigmoid()

        )
        self.culture_regressor = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 40),
            nn.BatchNorm1d(40),
            nn.Sigmoid()

        )

    def forward(self, inputs, batch=None):
        """
        call the modules in order, passing mha outputs upwards to the next tasks
        """

        assert isinstance(inputs, list) or isinstance(inputs, tuple), "Expected input to be a list or tuple with 5 elements (1 conv + 4 tf block activations"
        assert len(inputs) == 5

        # mappings
        conv_features = self.conv_mapping(inputs[0])
        voc_type_features = self.transformer_mappings[0](inputs[1])
        low_features = self.transformer_mappings[1](inputs[2])
        high_features = self.transformer_mappings[2](inputs[3])
        culture_emotion_features = self.transformer_mappings[3](inputs[4])

        # pass through mha
        voc_type_features = self.mhas[0](query=voc_type_features, key=conv_features, value=conv_features)[0]
        low_features = self.mhas[1](query=low_features, key=voc_type_features, value=voc_type_features)[0]
        high_features = self.mhas[2](query=high_features, key=low_features, value=low_features)[0]
        culture_emotion_features = self.mhas[3](query=culture_emotion_features, key=high_features, value=high_features)[0]

        # temporal pooling
        if self.params.pool == "avg":
            voc_type_features = torch.mean(voc_type_features, dim=1)
            low_features = torch.mean(low_features, dim=1)
            high_features = torch.mean(high_features, dim=1)
            culture_emotion_features = torch.mean(culture_emotion_features, dim=1)
        else:   # pick last time step
            voc_type_features = voc_type_features[:, -1, :]
            low_features = low_features[:, -1, :]
            high_features = high_features[:, -1, :]
            culture_emotion_features = culture_emotion_features[:, -1, :]

        # outputs
        voc_type_pred = self.voc_type_classifier(voc_type_features)
        low_pred = self.low_regressor(low_features)
        high_pred = self.high_regressor(high_features)
        culture_emotion_pred = self.culture_regressor(culture_emotion_features)

        return {"voc_type": voc_type_pred, "low": low_pred, "high": high_pred, "culture_emotion": culture_emotion_pred}


class TaskAttentionNetwork(nn.Module):
    """
    Task specific attention network, based on Liu 2019.
    """

    def __init__(self, num_layers, embedding_size, dropout=0.2) -> None:
        super().__init__()

        self.num_layers = num_layers    # number of layers of the backbone that give features

        self.mhas = nn.ModuleList(
            modules=[nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, dropout=dropout, batch_first=True) for _ in range(self.num_layers-1)]
        )

        self.lns = nn.ModuleList(
            modules=[nn.LayerNorm(embedding_size) for _ in range(self.num_layers -1)]
        )

        self.act = nn.GELU()

    def forward(self, inputs):
        """
        :input: list[B, L, D_emb]
        """
        assert len(inputs) == self.num_layers, "Expected {} inputs, got {}".format(len(inputs))

        # pass through mha + layernorm + gelu
        x = inputs[0]
        for i in range(self.num_layers -1):
            attn_out = self.mhas[i](query=inputs[i+1], key=x, value=x)[0]
            x = self.lns[i](attn_out + x)
            x = self.act(x)

        return x

            
        
        



class AttentionBranchModuleV2(nn.Module):
    """
    Branches out attention heads from the intermediate layers of the encoder network. Heads are connected in series, creating a parallel path to the backbone.
    Inspired by Song et al. (2022)
    """
    def __init__(self, params:Params, num_layers=4) -> None:
        super().__init__()
        self.params = params

        #conv_dim = 512
        tf_dim = 768
        embedding_size = params.embedding_size

        self.num_layers = num_layers
        self.dropout = 0.3

        # reduce input dimensionality to MHA
        self.transformer_mappings = nn.ModuleList(
            modules=[nn.Sequential(
                nn.Linear(tf_dim, embedding_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                )
            for _ in range(self.num_layers)] 
        )

        # mha
        #self.mhas = nn.ModuleList(
        #    modules=[nn.MultiheadAttention(embed_dim=embedding_size, num_heads=8, dropout=0.2, batch_first=True) for _ in range(self.num_layers-1)]
        #)

        # task attention networks - one per task
        self.tans = nn.ModuleList(modules=[
            TaskAttentionNetwork(self.num_layers, embedding_size, dropout=self.dropout) for _ in range(4)
        ])

         # output networks
        self.voc_type_classifier = nn.Sequential(
            #nn.BatchNorm1d(embedding_size),
            #nn.GELU(),
            nn.Linear(embedding_size, 8),
            nn.BatchNorm1d(8)

        )
        self.low_regressor = nn.Sequential(
            #nn.BatchNorm1d(embedding_size),
            #nn.GELU(),
            #TaskAttentionNetwork(self.num_layers, embedding_size, dropout=self.dropout),
            nn.Linear(embedding_size, 2),
            nn.BatchNorm1d(2)

        )
        self.high_regressor = nn.Sequential(
            #nn.BatchNorm1d(embedding_size),
            #nn.GELU(),
            #TaskAttentionNetwork(self.num_layers, embedding_size, dropout=self.dropout),
            nn.Linear(embedding_size, 10),
            nn.BatchNorm1d(10),
            nn.Sigmoid()

        )
        self.culture_regressor = nn.Sequential(
            #nn.BatchNorm1d(embedding_size),
            #nn.GELU(),
            #TaskAttentionNetwork(self.num_layers, embedding_size, dropout=self.dropout),
            nn.Linear(embedding_size, 40),
            nn.BatchNorm1d(40),
            nn.Sigmoid()

        )


    def forward(self, inputs, batch=None):
        """
        call the modules in order, passing mha outputs upwards to the next tasks
        """

        assert isinstance(inputs, list) or isinstance(inputs, tuple), "Expected input to be a list or tuple with 5 elements (1 conv + 4 tf block activations"
        assert len(inputs) == self.num_layers

        mapped_features = []
        for i, feat in enumerate(inputs):
            mf = self.transformer_mappings[i](feat)
            mapped_features.append(mf)

        # perform mha
        #voc_type_features = self.mhas[0](query=mapped_features[1], key=mapped_features[0], value=mapped_features[0])[0]
        #low_features = self.mhas[1](query=mapped_features[2], key=voc_type_features, value=voc_type_features)[0]
        #high_features = self.mhas[2](query=mapped_features[3], key=low_features, value=low_features)[0]
        #culture_emotion_features = self.mhas[3](query=mapped_features[4], key=high_features, value=high_features)[0]
        voc_type_features = self.tans[0](mapped_features)
        low_features = self.tans[1](mapped_features)
        high_features = self.tans[2](mapped_features)
        culture_emotion_features = self.tans[3](mapped_features)


        # temporal pooling
        if self.params.pool == "avg":
            voc_type_features = torch.mean(voc_type_features, dim=1)
            low_features = torch.mean(low_features, dim=1)
            high_features = torch.mean(high_features, dim=1)
            culture_emotion_features = torch.mean(culture_emotion_features, dim=1)
        else:   # pick last time step
            voc_type_features = voc_type_features[:, -1, :]
            low_features = low_features[:, -1, :]
            high_features = high_features[:, -1, :]
            culture_emotion_features = culture_emotion_features[:, -1, :]

        # outputs
        voc_type_pred = self.voc_type_classifier(voc_type_features)
        low_pred = self.low_regressor(low_features)
        high_pred = self.high_regressor(high_features)
        culture_emotion_pred = self.culture_regressor(culture_emotion_features)

        return {"voc_type": voc_type_pred, "low": low_pred, "high": high_pred, "culture_emotion": culture_emotion_pred}


class AbstractModel(nn.Module):

    """
    base class with feature extractor and classifier modules
    """

    def __init__(self, params:Params) -> None:
        super().__init__()
        self.params = params    # store these here

        self.feature_extractor = load_ssl_model(params)
        self.classifier = nn.Module()

    def forward(self, inputs, batch):
        """
        generic processing of fe features here?
        """
        if self.params.model.features == "attention":
            features =  self.feature_extractor(inputs)
        else:
            features = self.feature_extractor(inputs, return_hidden=True)

        return self.classifier(features)


    def freeze_fe(self):
        """
        Freezes the feature extractor layers
        """
        self.feature_extractor.requires_grad_(False)

    def unfreeze_fe(self):
        """
        Unfreezes the feature extractor layers
        """
        self.feature_extractor.requires_grad_(True)


class BaseMultitaskModel(AbstractModel):
    """
    Basic Multi Task learning (MTL)
    """

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        feat_dim = get_feature_dim(params)
        self.classifier = BaseMultiModule(feat_dim=feat_dim, params=params)

    def forward(self, inputs, batch):
        
        # feature extraction
        if self.params.features == "cnn": # pick only cnn features
            features = self.feature_extractor(inputs)
            features = features["extract_features"]
        elif self.params.features == "both":
            features = self.feature_extractor(inputs, return_hidden=True)
            features = torch.cat([features["last_hidden_state"], features["extract_features"]], dim=-1)
        else:   # only last layer 
            features = self.feature_extractor(inputs)
            features = features["last_hidden_state"]
        # classifier
        out = self.classifier(features, batch)

        return out

class StackModel(AbstractModel):
    """
    Stacked (Chained) Model that uses the predictions of one task for the next
    """

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        feat_dim = get_feature_dim(params)
        self.classifier = StackedModule(feat_dim=feat_dim, params=params)

    def forward(self, inputs, batch):
        # feature extraction
        if self.params.features == "cnn": # pick only cnn features
            features = self.feature_extractor(inputs)
            features = features["extract_features"]
        elif self.params.features == "both":
            features = self.feature_extractor(inputs, return_hidden=True)
            features = torch.cat([features["last_hidden_state"], features["extract_features"]], dim=-1)
        else:   # only last layer 
            features = self.feature_extractor(inputs)
            features = features["last_hidden_state"]
        # classifier
        out = self.classifier(features, batch)

        return out


class StackModelV2(AbstractModel):
    """
    Updated version of the Chain model which also chains the emotions within the tasks
    """
    def __init__(self, params: Params) -> None:
        super().__init__(params)

        feat_dim = get_feature_dim(params)
        self.classifier = StackedModuleV2(feat_dim=feat_dim, params=params)

    def forward(self, inputs, batch):
        # feature extraction
        if self.params.features == "cnn": # pick only cnn features
            features = self.feature_extractor(inputs)
            features = features["extract_features"]
        elif self.params.features == "both":
            features = self.feature_extractor(inputs, return_hidden=True)
            features = torch.cat([features["last_hidden_state"], features["extract_features"]], dim=-1)
        else:   # only last layer 
            features = self.feature_extractor(inputs)
            features = features["last_hidden_state"]
        # classifier
        out = self.classifier(features, batch)

        return out


class AttentionFusionModel(AbstractModel):
    """
    Model that fuses features from each task sequentially using attention
    """

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        self.feature_extractor_low = load_ssl_model(params)
        self.feature_extractor_high = load_ssl_model(params)
        self.feature_extractor_culture_emo = load_ssl_model(params)

        feat_dim = get_feature_dim(params)
        self.classifier = AttentionStackModule(feat_dim, params=params)

    def forward(self, inputs, batch=None):
        # ssl features
        voc_type_features = self.feature_extractor(inputs)
        low_features = self.feature_extractor_low(inputs)
        high_features = self.feature_extractor_high(inputs)
        culture_emotion_features = self.feature_extractor_culture_emo(inputs)

        if self.params.features == "both":
            voc_type_features = torch.cat([voc_type_features["extract_features"], voc_type_features["last_hidden_state"]], dim=-1)
            low_features = torch.cat([low_features["extract_features"], low_features["last_hidden_state"]], dim=-1)
            high_features = torch.cat([high_features["extract_features"], high_features["last_hidden_state"]], dim=-1)
            culture_emotion_features = torch.cat([culture_emotion_features["extract_features"], culture_emotion_features["last_hidden_state"]], dim=-1)
        else:
            if self.params.features == "cnn":
                key = "extract_features"
            else:
                key = "last_hidden_state"
            voc_type_features = voc_type_features[key]
            low_features = low_features[key]
            high_features = high_features[key]
            culture_emotion_features = culture_emotion_features[key]
                
        feats = [voc_type_features, low_features, high_features, culture_emotion_features]

        out = self.classifier(feats, batch=batch)

        return out


class AttentionBranchModel(AbstractModel):
    """
    Model whose features branch out from intermediate layers of the backbone feature extractor and and passed through a sequence of MHA
    """
    def __init__(self, params: Params) -> None:
        super().__init__(params)

        # select features to branch out
        branch_layers = self.params.branch_layers
        if branch_layers == "first":
            self.layer_indices= [1, 2, 3, 4]
        elif branch_layers == "middle":
            self.layer_indices=  [5, 6, 7, 8]
        elif branch_layers == "last":
            self.layer_indices=  [9, 10, 11, 12]
        else:
            self.layer_indices=  [3, 6, 9, 12]

        self.classifier = AttentionBranchModule(params=self.params)


    def forward(self, inputs, batch=None):
        # get the features
        features = self.feature_extractor(inputs, output_hidden_states=True)
        # assemble input to branch (conv features + 4 hidden transformer states)
        branch_inputs = [features["extract_features"]]
        for i in self.layer_indices:
            branch_inputs.append(features["hidden_states"][i])

        out = self.classifier(branch_inputs)

        return out


class AttentionBranchModelV2(AbstractModel):
    """
    Model whose features branch out from intermediate layers of the backbone feature extractor and and passed through a sequence of MHA
    """
    def __init__(self, params: Params) -> None:
        super().__init__(params)

        # select features to branch out
        branch_layers = self.params.branch_layers
        
        if branch_layers == "first":
            self.layer_indices= [3, 6, 9, 12]
        elif branch_layers == "middle":
            self.layer_indices=  [6, 8, 10, 12]
        elif branch_layers == "last":
            self.layer_indices=  [9, 10, 11, 12]
        else:
            self.layer_indices=  [2, 4, 6, 8, 10, 12]

        self.classifier = AttentionBranchModuleV2(params=self.params, num_layers=len(self.layer_indices))


    def forward(self, inputs, batch=None):
        # get the features from the deep network
        features = self.feature_extractor(inputs, output_hidden_states=True)

        # assemble input to branch (conv features + 4 hidden transformer states)
        #branch_inputs = [features["extract_features"]]
        
        branch_inputs = []
        for i in self.layer_indices:
            branch_inputs.append(features["hidden_states"][i])

        out = self.classifier(branch_inputs)

        return out
        

def model_factory(model_params:Params) -> AbstractModel:
    """
    Factory method that builds a model from the catalogue based on the specified params
    :model_params a Params object containing model architecture info
    :returns: A subclass of Abstractmodel
    """

    if model_params.model_name == "basemtl":
        return BaseMultitaskModel(model_params)
    elif model_params.model_name == "stacked":
        return StackModel(model_params)
    elif model_params.model_name == "stacked_v2":
        return StackModelV2(model_params)
    elif model_params.model_name == "attnfusion":
        return AttentionFusionModel(model_params)
    elif model_params.model_name == "attnbranch":
        return AttentionBranchModel(model_params)
    elif model_params.model_name == "attnbranch_v2":
        return AttentionBranchModelV2(model_params)
    else:
        raise ValueError("Model architecture {} is not recognised".format(model_params.model_name))



            

        
        


    