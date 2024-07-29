import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torch import nn
import numpy as np
import re
import os
import PIL
from pathlib import Path


MODEL_PATH = (
    Path(__file__).parent.joinpath("resources/vae-oid.npz").resolve().as_posix()
)

EMBEDDING_DIM = 512
NUM_EMBEDDINGS = 128
NUM_RESNET_BLOCKS = 2

SEGMENT_DETECT_RE = re.compile(
    r"(.*?)"
    + r"<loc(\d{4})>" * 4
    + r"\s*"
    + "(?:%s)?" % (r"<seg(\d{3})>" * 16)
    + r"\s*([^;<>]+)? ?(?:; )?",
)

SEG_TOKENS = np.array(["<seg%03d>" % i for i in range(128)])
LOC_TOKENS = np.array(["<loc%04d>" % i for i in range(1024)])


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        num_downsample_layers = 4
        layer_num = num_downsample_layers - 1
        dim = 16
        channels = 1
        self.embedding_dim = EMBEDDING_DIM

        enc_layers = [nn.Conv2d(channels, dim, 4, stride=2, padding=1), nn.ReLU()]

        for i in range(layer_num):
            enc_layers.append(nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            dim = dim * 2

        for i in range(NUM_RESNET_BLOCKS):
            enc_layers.append(ResBlock(dim))
        enc_layers.append(nn.Conv2d(dim, EMBEDDING_DIM, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.embedding = nn.Embedding(NUM_EMBEDDINGS, self.embedding_dim)

    def norm(self, x):
        return 2.0 * (x - 0.5)

    def forward(self, img):
        img = img.float()
        img = self.norm(img)

        inputs = self.encoder(img)

        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 128
        num_upsample_layers = 4
        layer_num = num_upsample_layers - 1
        channels = 1

        self.num_embeddings = NUM_EMBEDDINGS
        self.embedding_dim = EMBEDDING_DIM

        dec_layers = [nn.Conv2d(EMBEDDING_DIM, dim, 1), nn.ReLU()]

        for i in range(NUM_RESNET_BLOCKS):
            dec_layers.append(ResBlock(dim))

        dec_layers.append(nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1))
        dec_layers.append(nn.ReLU())
        for i in range(layer_num):
            dec_layers.append(nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1))
            dec_layers.append(nn.ReLU())
            dim = dim // 2
        dec_layers.append(nn.Conv2d(dim, channels, 1))
        self.decoder = nn.Sequential(*dec_layers)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, encoding_indices):
        quantized = self.embedding.weight[encoding_indices].clone()
        out = self.decoder(quantized.t().view([1, EMBEDDING_DIM, 4, 4]))
        return out


class SegmentationTokenizer:

    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            np_state_dict = dict(np.load(f))

        encoder_state_dict = {}

        for key, weight in np_state_dict.items():
            if "encoder." in key:
                encoder_state_dict[key] = torch.tensor(weight)

            if "embedding" in key:
                encoder_state_dict["embedding.weight"] = torch.tensor(weight)

        self.encoder_model = Encoder()
        self.encoder_model.load_state_dict(encoder_state_dict)

        decoder_state_dict = {}

        for key, weight in np_state_dict.items():
            if "decoder." in key:
                decoder_state_dict[key] = torch.tensor(weight)

            if "embedding" in key:
                decoder_state_dict["embedding.weight"] = torch.tensor(weight)

        self.decoder_model = Decoder()
        self.decoder_model.load_state_dict(decoder_state_dict)

    def encode(self, image, xyxy_bboxes, masks, classes):
        w, h = image.size

        suffix_components = []
        for xyxy, mask, class_name in zip(xyxy_bboxes, masks, classes):
            xyxy = np.array(xyxy)
            mask = np.array(mask)

            y1 = xyxy[1].astype(np.int32)
            x1 = xyxy[0].astype(np.int32)
            y2 = xyxy[3].astype(np.int32)
            x2 = xyxy[2].astype(np.int32)

            mask = torch.tensor(mask.astype(np.uint8), dtype=torch.uint8)
            mask = resize(
                # mask[None, y1:y2, x1:x2, None],
                mask[y1:y2, x1:x2].unsqueeze(0),
                [64, 64],
                # interpolation="bilinear",
                antialias=True,
            )

            mask_indices = self.encoder_model(mask.unsqueeze(0)).numpy()
            mask_string = np.take(SEG_TOKENS, mask_indices)

            bbox = xyxy[[1, 0, 3, 2]] / np.array([h, w, h, w])
            binned_loc = (bbox * 1023).astype(np.int32)
            binned_loc = np.clip(binned_loc, 0, 1023)
            loc_string = np.take(LOC_TOKENS, binned_loc)

            suffix_part = "".join(np.concatenate([loc_string, mask_string]).tolist())
            suffix_part = f"{suffix_part} {class_name}"
            suffix_components.append(suffix_part)

        suffix = " ; ".join(suffix_components)
        return suffix

    def decode(self, text, width, height, unique_labels=False):
        """Returns objs for a string with "<loc>" and "<seg>" tokens."""
        objs = []
        seen = set()
        while text:
            m = SEGMENT_DETECT_RE.match(text)
            if not m:
                break
            gs = list(m.groups())

            before = gs.pop(0)
            name = gs.pop()
            y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]

            y1, x1, y2, x2 = map(
                round, (y1 * height, x1 * width, y2 * height, x2 * width)
            )
            seg_indices = gs[4:20]
            if seg_indices[0] is None:
                mask = None
            else:
                seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)

                seg_indices = torch.tensor(seg_indices, dtype=torch.int64)

                m64 = self.decoder_model(seg_indices).squeeze().detach().numpy()
                m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
                m64 = PIL.Image.fromarray((m64 * 255).astype("uint8"))

                mask = np.zeros([height, width])
                if y2 > y1 and x2 > x1:
                    mask[y1:y2, x1:x2] = (
                        np.array(m64.resize([x2 - x1, y2 - y1])) / 255.0
                    )

            content = m.group()
            if before:
                objs.append(dict(content=before))
                content = content[len(before) :]
            while unique_labels and name in seen:
                name = (name or "") + "'"
            seen.add(name)
            objs.append(
                dict(content=content, xyxy=(x1, y1, x2, y2), mask=mask, name=name)
            )
            text = text[len(before) + len(content) :]

        if text:
            objs.append(dict(content=text))

        return objs

