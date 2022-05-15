import torch
import torch.nn as nn
from torch.nn import functional


class TransH(nn.Module):
    """

    """

    def __init__(self, entity_number: int, relation_number: int, dimension: int = 64,
                 gamma: float = 1.0, c: float = 1.0, epsilon: float = 1e-5, device='cpu'):
        """
        Parameters
            ----------
            entity_number :
                количество сущностей
            relation_number :
                количество отношений
            dimension :
                размер эмбеддинга. The default is 64.
            gamma :
                гиперпараметр гамма в формуле 7. The default is 1.0.
                при уменьшении ошибки на положительных триплетах, общая ошибка scale_loss становится
                становится нулем, но дополнительно мы хотим увеличить расстояние между отрицательными примерами,
                поэтому мы делаем "зазор" (margin). Без этого зазора при условии, что ошибка на положительных
                примерах нулевая (что мы и хотим) ошибка на отрицательных примерах не будет становится больше нуля
                т.к. ReLu для первого слагаемого делает ошибку нулевой, не давая дифферециироваться.
            c :
                гиперпараметр, взвешивающий важность мягких ограничений. The default is 1.0.
                добавляет больший вес ошибкам, возникающим в следствии нарушения условий.
            epsilon :
                параметр используемый в условии ортогональности нормали и гиперплоскости H. The default is 1e-5.


        """
        super(TransH, self).__init__()
        self.dimension = dimension
        self.gamma = gamma
        self.c = c
        self.entity_number = entity_number
        self.relation_number = relation_number
        self.epsilon = torch.FloatTensor([epsilon]).to(device)

        # эмбеддинг вектор нормали гиперплоскости на которую проецируем сущности
        self.w_r_emb = torch.nn.Embedding(num_embeddings=relation_number,
                                          embedding_dim=dimension)
        # эмбеддинг векторов отношений на гиперплоскости
        # вектор отношений принадлежит гиперплоскости т.к. является ортогональным нормали по свойству 8,
        # проходит через 0, в свою очередь гиперплоскость обладает теми же свойствами
        self.d_r_emb = torch.nn.Embedding(num_embeddings=relation_number,
                                          embedding_dim=dimension)

        # эмбеддинг сущностей
        self.entity_embedding = torch.nn.Embedding(num_embeddings=entity_number,
                                                   embedding_dim=dimension)

        # создаем случайные веса нужной нам размерности.
        nn.init.xavier_uniform_(self.w_r_emb.weight.data)
        nn.init.xavier_uniform_(self.d_r_emb.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)

    def load_trained_embeddings(self, entity_vector: torch.Tensor,
                                relation_vector: torch.Tensor,
                                relation_normal: torch.Tensor):
        """
        Импорт уже обученных векторов
        Parameters
        ----------
        entity_vector : тензор эмбеддингов сущностей
        relation_vector : тензор эмбеддингов отношений на гиперплолскости
        relation_normal : тензор эмбеддингов нормалей гиперплоскостей
        """
        self.entity_embedding.weight.data = entity_vector
        self.d_r_emb.weight.data = relation_vector
        self.w_r_emb.weight.data = relation_normal

    @staticmethod
    def projected(entity: torch.Tensor, w_r: torch.Tensor):
        """
        теоретическая справка, формулы 3,4
        Parameters
        ----------
        entity: torch.Tensor
            сущности
        w_r: torch.Tensor

        Returns
        -------
        вектор проекции на гиперпроскость
        """
        # dim=-1 - нормализация по последнему измерению,
        # w_r  проецируется на единичный шар ф-ия проекции векторов на гиперплоскость
        w_r = functional.normalize(w_r, dim=-1)
        return entity - torch.sum(entity * w_r, dim=1, keepdim=True) * w_r

    def distance(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
        """
        функция оценки f_r(h,t) (теоретическая справка,формула 5)
        Parameters
        ----------
        h : torch.Tensor
            тензор состоящий из id головных сущностей
        r : torch.Tensor
            тензор состоящий из id отношений
        t : torch.Tensor
            тензор состоящий из id хвостовых сущностей

        Returns
        -------
        значение функции оценки
        """

        head = self.entity_embedding(h)
        w_r = self.w_r_emb(r)
        d_r = self.d_r_emb(r)
        tail = self.entity_embedding(t)

        head_hyper = self.projected(head, w_r)
        tail_hyper = self.projected(tail, w_r)

        distance = head_hyper + d_r - tail_hyper
        score = torch.norm(distance, dim=1)
        return score

    @staticmethod
    def scale_loss(embedding: torch.Tensor):
        """
        ошибка основанная на ограничении 7. Все эмбеддинги находятся в единичном шаре
        Parameters
        ----------
        embedding : torch.Tensor

        Returns
        -------
         функция ошибки
        """
        return torch.sum(torch.relu(torch.norm(embedding, dim=1) - 1))

    # ошибка основанная на ограничении 8: вектор перемещения находится на гиперплоскости.

    def orthogonal_loss(self, relation_embedding: torch.Tensor, w_embedding: torch.Tensor):
        """
        ошибка основанная на ограничении 8: вектор перемещения находится на гиперплоскости.
        Parameters
        ----------
        relation_embedding : torch.Tensor
        w_embedding : torch.Tensor

        Returns
        -------
        функция ошибки
        """
        dot = torch.sum(relation_embedding * w_embedding, dim=1) ** 2
        norm = torch.norm(relation_embedding, dim=1) ** 2
        loss = torch.sum(
            torch.relu(dot / norm - self.epsilon ** 2)
        )
        return loss

    def loss(self, positive_triplets: torch.Tensor, negative_triplets: torch.Tensor):
        """
        функция потерь с мягкими ограничениями. Теоретическая справка формула 10
        к основной функции потерь прибавляется с весом С ошибки в случае не выпонения ограничений (7, 8)
        Parameters
        ----------
        positive_triplets : torch.Tensor
        negative_triplets : torch.Tensor

        Returns
        -------
        возвращает функцию потерь с мягкими ограничениями
        """

        h, r, t = torch.chunk(positive_triplets, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(negative_triplets, 3, dim=1)

        positive = self.distance(h, r, t)
        negative = self.distance(h_c, r_c, t_c)

        loss = torch.relu(positive - negative + self.gamma).mean()

        entity_embedding = self.entity_embedding(torch.cat([h, t, h_c, t_c]))
        relation_embedding = self.d_r_emb(torch.cat([r, r_c]))
        w_embedding = self.w_r_emb(torch.cat([r, r_c]))

        orthogonal_loss = self.orthogonal_loss(relation_embedding, w_embedding)

        scale_loss = self.scale_loss(entity_embedding)

        return loss + self.c * (scale_loss / len(entity_embedding) + orthogonal_loss / len(relation_embedding))
