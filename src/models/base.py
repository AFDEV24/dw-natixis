from dataclasses import dataclass


@dataclass
class FundMetadata:
    region: str
    fund_name: str
    date: str
