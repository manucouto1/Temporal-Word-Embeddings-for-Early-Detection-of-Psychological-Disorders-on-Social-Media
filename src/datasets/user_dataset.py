import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer


class UserDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el dataset de usuarios.
        Args:
            data (pd.DataFrame): DataFrame con al menos las columnas 'user_id', 'chunk_id', 'text', 'label'.
                                 Se espera que 'text' contenga el texto real.
                                 'chunk_id' y 'user_id' se usan para agrupar.
        """
        self.data = data
        self.users = data["user"].unique()
        self.user_data = {}

        # Agrupar los datos por usuario y chunk
        for user_id in self.users:
            user_df = self.data[self.data["user"] == user_id]
            user_label = user_df["label"].values.tolist()

            # Agrupar los textos por chunk para este usuario
            chunks_list = []
            if not user_df.empty:
                for chunk in user_df.text.tolist():
                    # for chunk_id in sorted(user_df["chunk"].unique()):
                    #     texts_in_chunk = user_df[user_df["chunk"] == chunk_id][
                    #         "text"
                    #     ].tolist()
                    chunks_list.append(chunk)

            self.user_data[user_id] = (chunks_list, user_label)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        return self.user_data[user_id]


class CustomCollateFn:
    def __init__(self, tokenizer_name: str, max_seq_length: int, device: torch.device):
        """
        Inicializa el colator con un tokenizador y parámetros de padding.
        Args:
            tokenizer_name (str): Nombre del modelo de tokenizador (ej. 'bert-base-uncased').
            max_seq_length (int): Longitud máxima de secuencia para el tokenizador.
            device (torch.device): Dispositivo (CPU/GPU) donde mover los tensores resultantes.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.device = device
        self.fixed_num_chunks = (
            10  # <--- ¡Aquí definimos el número fijo de grupos/chunks!
        )

    def __call__(self, batch: list):
        """
        Este método es llamado por el DataLoader para procesar un lote de datos.
        Args:
            batch (list): Una lista de elementos obtenidos del __getitem__ de tu Dataset.
                          Cada elemento es (user_chunks_list, label).
        Returns:
            dict: Un diccionario de tensores listos para el modelo, incluyendo la información
                  para la reconstrucción jerárquica.
        """

        all_texts_for_tokenization = []  # Lista final aplanada de todas las cadenas de texto
        user_labels = []

        # Lista para almacenar el número de textos en cada chunk (aplanada para todo el lote)
        flat_chunk_text_counts = []

        # Para mapear cada texto plano a su ID de chunk plano
        flat_text_to_flat_chunk_map = []

        # Para mapear cada chunk plano a su ID de usuario en el lote
        flat_chunk_to_batch_user_map = []

        global_flat_chunk_idx = 0  # ID único para cada chunk en todo el lote

        # Iterar sobre cada usuario en el lote actual
        for user_batch_idx, (user_chunks_list_original, label) in enumerate(batch):
            user_labels.append(label)

            # --- Lógica de padding/truncado para asegurar 10 grupos (chunks) por usuario ---
            # 1. Truncar: Si el usuario tiene más de `fixed_num_chunks` chunks, nos quedamos con los primeros.
            user_chunks_list_processed = user_chunks_list_original[
                : self.fixed_num_chunks
            ]

            # 2. Paddear: Si el usuario tiene menos de `fixed_num_chunks` chunks, añadimos chunks vacíos.
            num_actual_chunks = len(user_chunks_list_processed)
            padding_needed_chunks = self.fixed_num_chunks - num_actual_chunks

            if padding_needed_chunks > 0:
                # Un chunk de padding es una lista vacía de textos ([])
                user_chunks_list_processed.extend([[]] * padding_needed_chunks)

            # Ahora `user_chunks_list_processed` SIEMPRE tiene `self.fixed_num_chunks` elementos.

            # --- Procesamiento de los chunks (ahora siempre 10) ---
            for chunk_in_user_idx, texts_in_chunk in enumerate(
                user_chunks_list_processed
            ):
                # Almacenar la longitud de los textos en este chunk (0 si es un chunk de padding)
                flat_chunk_text_counts.append(len(texts_in_chunk))

                # Mapear cada texto de este chunk a su ID de chunk plano (solo para textos no vacíos)
                flat_text_to_flat_chunk_map.extend(
                    [global_flat_chunk_idx] * len(texts_in_chunk)
                )

                # Añadir todos los textos a la lista principal para la tokenización de BERT
                all_texts_for_tokenization.extend(
                    [str(text) for text in texts_in_chunk]
                )

                # Mapear cada chunk plano a su ID de usuario en el lote
                flat_chunk_to_batch_user_map.append(user_batch_idx)

                global_flat_chunk_idx += 1  # Incrementar el ID único del chunk

        # Convertir etiquetas a tensor
        labels_tensor = torch.tensor(user_labels, dtype=torch.float32).to(self.device)

        # Tokenizar todas las cadenas de texto aplanadas para BERT
        encoded_inputs = self.tokenizer(
            all_texts_for_tokenization,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_attention_mask=True,
        ).to(self.device)

        return {
            "input_ids": encoded_inputs[
                "input_ids"
            ],  # Forma: (Total_Texts_in_Batch, T_max)
            "attention_mask": encoded_inputs[
                "attention_mask"
            ],  # Forma: (Total_Texts_in_Batch, T_max)
            "labels": labels_tensor,  # Forma: (Batch_Size_Users,)
            # Información para la reconstrucción del nivel de Chunk (Textos por Chunk)
            "flat_chunk_text_counts": torch.tensor(
                flat_chunk_text_counts, dtype=torch.long
            ).to(self.device),  # Forma: (Total_Chunks_in_Batch,)
            "flat_text_to_flat_chunk_map": torch.tensor(
                flat_text_to_flat_chunk_map, dtype=torch.long
            ).to(self.device),  # Forma: (Total_Texts_in_Batch,)
            # Información para la reconstrucción del nivel de Usuario (Chunks por Usuario - AHORA FIJO)
            "flat_chunk_to_batch_user_map": torch.tensor(
                flat_chunk_to_batch_user_map, dtype=torch.long
            ).to(self.device),  # Forma: (Total_Chunks_in_Batch,)
            # Dimensiones totales del lote para verificaciones
            "total_texts_in_batch": len(
                all_texts_for_tokenization
            ),  # Suma de textos reales
            "total_chunks_in_batch": global_flat_chunk_idx,  # Esto será `Batch_Size_Users * self.fixed_num_chunks`
            "batch_size_users": len(batch),
        }
