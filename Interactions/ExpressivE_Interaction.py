import torch
from pykeen.nn.modules import Interaction

from Utils import preprocess_relations, preprocess_entities


class ExpressivE_Interaction(Interaction):
    # dimensioni delle embedding: handled via *_representations_kwargs nel modello
    relation_shape = 'e'
    entity_shape = 'd'

    def __init__(self, p: int, tanh_map: bool = True, min_denom: float = 0.5):
        super().__init__()
        self.p = p  # Norm that shall be used (either 1 or 2)
        self.tanh_map = tanh_map
        self.min_denom = min_denom

    def distance(self, d_h, d_t, c_h, c_t, s_h, s_t, h, t):
        # Calculate the distance of the triple

        # concatena distanze, centri e slopes
        d = torch.cat([d_h, d_t], dim=-1)
        c = torch.cat([c_h, c_t], dim=-1)
        s = torch.cat([s_t, s_h], dim=-1)

        # broadcast e concat dei vettori di entità
        h_b, t_b = torch.broadcast_tensors(h, t)
        ht = torch.cat([h_b, t_b], dim=-1)
        th = torch.cat([t_b, h_b], dim=-1)

        contextualized_pos = torch.abs(ht - c - s * th)

        is_entity_pair_within_para = contextualized_pos.le(d).all(dim=-1)

        w = 2 * d + 1

        # Case 1: Triple outside of Para
        k = 0.5 * (w - 1) * (w - 1 / w)
        dist = contextualized_pos * w - k

        # Case 2: Triple within Para
        dist[is_entity_pair_within_para] = (contextualized_pos / w)[is_entity_pair_within_para]

        return dist

    def get_score(self, d_h, d_t, c_h, c_t, s_h, s_t, h, t):
        dist = self.distance(d_h, d_t, c_h, c_t, s_h, s_t, h, t)
        return -dist.norm(p=self.p, dim=-1)

    def forward(self, h, r, t):
        # Preprocessing delle relazioni ed entità
        d_h, d_t, c_h, c_t, s_h, s_t = preprocess_relations(
            r, tanh_map=self.tanh_map, min_denom=self.min_denom
        )
        h, t = preprocess_entities([h, t], tanh_map=self.tanh_map)

        return self.get_score(d_h, d_t, c_h, c_t, s_h, s_t, h, t)
