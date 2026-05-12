from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class GCPConfig:
    project: str
    location: str


@dataclass
class LLMConfig:
    model: str
    temperature: float


@dataclass
class PlacesConfig:
    max_candidates: int
    nearby_search_url: str
    geocode_url: str


@dataclass
class RecallConfig:
    default_radius_meters: int
    max_radius_meters: int


@dataclass
class EmbeddingConfig:
    model: str
    dimensions: int
    top_k: int


@dataclass
class ServingConfig:
    variant_id: str


@dataclass
class ResultsConfig:
    default_top_k: int


@dataclass
class Config:
    gcp: GCPConfig
    llm: LLMConfig
    places: PlacesConfig
    recall: RecallConfig
    embedding: EmbeddingConfig
    serving: ServingConfig
    results: ResultsConfig


def load_config(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        gcp=GCPConfig(**raw["gcp"]),
        llm=LLMConfig(**raw["llm"]),
        places=PlacesConfig(**raw["places"]),
        recall=RecallConfig(**raw["recall"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        serving=ServingConfig(**raw["serving"]),
        results=ResultsConfig(**raw["results"]),
    )


cfg = load_config()
