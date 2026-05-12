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
    primary_type: Optional[str] = None
    reviews: list[str] = []   # up to 5 snippets from Places API (Preferred SKU)


class NearbySearchParams(BaseModel):
    location_name: str
    latitude: float
    longitude: float
    radius_meters: int
    included_types: list[str]
    keyword: Optional[str]
    max_results: int


class UserIntent(BaseModel):
    """All structured inputs collected from the question phase."""
    query: str
    min_rating: float
    max_rating: float
    intent_text: str
    selected_types: list[str]
    city: str
    country: str
    description: str   # Q4: free-text ideal place description, embedded for semantic ranking

    @property
    def location_name(self) -> str:
        return f"{self.city}, {self.country}"


class Recommendation(BaseModel):
    rank: int
    place: Place
    score: float
    reason: str
