from pydantic import BaseModel
from typing import Optional


class Place(BaseModel):
    name: str
    address: str
    rating: Optional[float] = None
    user_rating_count: Optional[int] = None
    price_level: Optional[str] = None
    maps_url: Optional[str] = None
    open_now: Optional[bool] = None


class SearchParams(BaseModel):
    search_query: str
    reasoning: str


class Recommendation(BaseModel):
    rank: int
    place: Place
    score: float
    reason: str
