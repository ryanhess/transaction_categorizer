from fastapi import FastAPI
from inference import cat
from models import TransactionRequest, TransactionResponse
from dataclasses import dataclass

# app = FastAPI()

txns = [
    TransactionRequest(id=1, payee="HAZUKI SUSHI", outflow=95.86),
    TransactionRequest(id=2, payee="Transfer : Main Savings 5886", outflow=300.00),
    TransactionRequest(id=3, payee="CHEWY.COM", outflow=53.03),
    TransactionRequest(id=4, payee="AMO SEAFOOD", outflow=100.41),
    TransactionRequest(id=5, payee="Stephen Pruden D.C.", outflow=220.00),
    TransactionRequest(id=6, payee="ParkWhiz, Inc.", outflow=21.20),
    TransactionRequest(id=7, payee="Amazon", outflow=44.85),
    TransactionRequest(id=8, payee="SAKURA RAMEN HOUSE", outflow=14.50),
    TransactionRequest(id=9, payee="BROOKLYN DINER", outflow=32.00),
    TransactionRequest(id=10, payee="H Mart", outflow=67.80),
    TransactionRequest(id=11, payee="CONTAINERSTOREWESTBURY", outflow=89.50),
    TransactionRequest(id=12, payee="OK PETROLEUM", outflow=45.00),
    TransactionRequest(id=13, payee="Transfer : Home Escrow 1597", outflow=157.86),
    TransactionRequest(id=14, payee="The Home Depot #1213", outflow=67.23),
    TransactionRequest(id=15, payee="Solid State Coffee", outflow=6.75),
    TransactionRequest(id=16, payee="STOP 1 BAGEL& DELI", outflow=12.40),
    TransactionRequest(id=17, payee="National Grid", outflow=135.30),
    TransactionRequest(id=18, payee="LIPA", outflow=130.48),
    TransactionRequest(id=19, payee="Transfer : Autopay Bills 6671", outflow=620.00),
    TransactionRequest(id=20, payee="TST* GOLDEN OAK BISTRO", outflow=42.50),
    TransactionRequest(id=21, payee="SORENSON LUMBER INC", outflow=234.56),
    TransactionRequest(id=22, payee="MTA*LIRR STATION TIX", outflow=18.75),
    TransactionRequest(id=23, payee="NASSAU MEAT MARKET INC", outflow=24.99),
    TransactionRequest(id=24, payee="Transfer : Marzena 7072", outflow=3500.00),
    TransactionRequest(id=25, payee="NORTH SHORE THAI KITCHEN", outflow=14.20),
    TransactionRequest(id=26, payee="GLEN HEAD HARDWARE", outflow=28.90),
    TransactionRequest(id=27, payee="State Farm", outflow=175.00),
    TransactionRequest(id=28, payee="AMAGANSETT IGA", outflow=112.45),
    TransactionRequest(id=29, payee="Transfer : Gabe Savings 5665", outflow=1103.29),
    TransactionRequest(id=30, payee="MANGO GRILL & TACO BAR", outflow=48.30),
    TransactionRequest(id=31, payee="Gabriel Hurtado", inflow=103.29),
    TransactionRequest(id=32, payee="AUTUMN LEAVES USED BOOKS", outflow=16.78),
    TransactionRequest(id=33, payee="TST* HARBOR VIEW CAFE", outflow=38.95),
    TransactionRequest(id=34, payee="GEICO", outflow=175.00),
    TransactionRequest(id=35, payee="Transfer : Home Improvement 3617", outflow=119.08),
    TransactionRequest(id=36, payee="OYSTER BAY BREWING CO", outflow=42.30),
    TransactionRequest(id=37, payee="PARKS AND RECREATION", outflow=24.99),
    TransactionRequest(id=38, payee="Grow Organic", outflow=35.60),
    TransactionRequest(id=39, payee="ADVANCE AUTO PARTS #7115", outflow=29.99),
    TransactionRequest(id=40, payee="Transfer : Vectorworks 2463", outflow=83.22),
    TransactionRequest(id=41, payee="JADE PALACE SZECHUAN", outflow=12.50),
    TransactionRequest(id=42, payee="OLD STONE MARKET", outflow=98.34),
    TransactionRequest(id=43, payee="ABEETZA PIZZA", outflow=16.25),
    TransactionRequest(id=44, payee="GRAND BRASS", outflow=59.00),
    TransactionRequest(id=45, payee="ROSLYN PIZZA & PASTA", outflow=18.99),
    TransactionRequest(id=46, payee="Transfer : Main Savings 5886", outflow=8802.39),
    TransactionRequest(id=47, payee="Dil-e Punjab Deli", outflow=13.45),
    TransactionRequest(id=48, payee="Amazon.com*RM99B6YC0", outflow=149.99),
    TransactionRequest(id=49, payee="MAIDSTONE MARKET & DELI", outflow=33.21),
    TransactionRequest(id=50, payee="UMAMI BURGER GARDEN CITY", outflow=16.80),
    TransactionRequest(
        id=51, payee="Transfer : Ally Interest Checking", outflow=104.37
    ),
    TransactionRequest(id=52, payee="Cumberland Farms", outflow=15.60),
    TransactionRequest(id=53, payee="CRAFTERS GALLERY", outflow=42.15),
    TransactionRequest(id=54, payee="TST* FIRESIDE TAVERN", outflow=35.80),
    TransactionRequest(id=55, payee="CARDIOVASCULAR WELLNESS L", outflow=210.00),
    TransactionRequest(id=56, payee="Apple", outflow=0.99),
    TransactionRequest(id=57, payee="LOCUST VALLEY PIZZA CAFE", outflow=22.75),
    TransactionRequest(id=58, payee="Poshmark", outflow=34.80),
    TransactionRequest(id=59, payee="Transfer : Main Savings 5886", outflow=124.11),
    TransactionRequest(id=60, payee="CEDAR CREEK BAR AND GRILL", outflow=56.80),
    TransactionRequest(id=61, payee="COMMUNITY CRAFT", outflow=45.67),
    TransactionRequest(id=62, payee="TOYOTA TIS TECH SERV", outflow=69.99),
    TransactionRequest(id=63, payee="TST* CARISSAS BAKERY - 22", outflow=12.40),
    TransactionRequest(id=64, payee="1Password", outflow=5.99),
    TransactionRequest(id=65, payee="TWO GUYS WINE & LIQUOR", outflow=28.50),
    TransactionRequest(id=66, payee="NYC BAGEL & COFFEE HOUSE", outflow=14.25),
    TransactionRequest(id=67, payee="GEP Payroll Svcs", inflow=3097.83),
    TransactionRequest(id=68, payee="SHAKE SHACK GARDEN CITY", outflow=19.35),
    TransactionRequest(id=69, payee="Transfer : Home Escrow 1597", outflow=6484.05),
    TransactionRequest(id=70, payee="CAMPGROUND BEER MARKE", outflow=18.75),
    TransactionRequest(id=71, payee="FLOUR POWER BAKERY", outflow=11.50),
    TransactionRequest(id=72, payee="GEP Payroll Svcs", inflow=1921.18),
    TransactionRequest(id=73, payee="TST* SALT & VINE", outflow=37.89),
    TransactionRequest(id=74, payee="ALLPAID*Village Of Mineol", outflow=79.99),
    TransactionRequest(id=75, payee="Dermstore", outflow=33.44),
    TransactionRequest(id=76, payee="Transfer : Marzena 7072", inflow=30.00),
    TransactionRequest(id=77, payee="THE COFFEE SHOP", outflow=5.45),
    TransactionRequest(id=78, payee="ACE HARDWARE OF GLEN C", outflow=45.67),
    TransactionRequest(id=79, payee="A RAZZANO`S IMPORTED FOOD", outflow=26.99),
    TransactionRequest(id=80, payee="Robinhood", outflow=110.00),
    TransactionRequest(id=81, payee="SEAMLSSGOLDENDRAGONCH", outflow=22.50),
    TransactionRequest(id=82, payee="Transfer : Autopay Bills 6671", outflow=450.00),
    TransactionRequest(id=83, payee="KENCO THE WORK & PLAY OUT", outflow=62.30),
    TransactionRequest(id=84, payee="PORTSIDE SEAFOOD GRILL", outflow=47.60),
    TransactionRequest(id=85, payee="Interest", inflow=12.53),
    TransactionRequest(id=86, payee="Wines on Broadway", outflow=29.25),
    TransactionRequest(id=87, payee="SP WANDP.COM", outflow=54.99),
    TransactionRequest(id=88, payee="SUNRISE GLEN HEAD", outflow=8.10),
    TransactionRequest(id=89, payee="Transfer : Gabe Savings 5665", outflow=500.00),
    TransactionRequest(id=90, payee="Mountain Rose Herbs", outflow=39.99),
    TransactionRequest(id=91, payee="GEP Payroll Svcs", inflow=3097.83),
    TransactionRequest(id=92, payee="PY *MAIN STREET FARM", outflow=41.20),
    TransactionRequest(id=93, payee="SEAMLSSLUCKYNOODLEBAR", outflow=18.50),
    TransactionRequest(id=94, payee="ABOFFS PAINTS - 12 - GLEN", outflow=42.15),
    TransactionRequest(id=95, payee="TST* ROSEMARY'S KITCHEN", outflow=28.90),
    TransactionRequest(id=96, payee="Transfer : Home Improvement 3617", outflow=200.00),
    TransactionRequest(id=97, payee="Gabriel Hurtado", inflow=1000.00),
    TransactionRequest(id=98, payee="SYLVAN SPIRITS", outflow=31.45),
    TransactionRequest(id=99, payee="Audible", outflow=14.95),
    TransactionRequest(id=100, payee="Cash Withdrawal at Branch", outflow=300.00),
]


@dataclass
class CompleteTxn:
    id: int
    payee: str
    inflow: float
    outflow: float
    category: str


results = cat.predict(txns)

complete = [
    CompleteTxn(
        id=req.id,
        payee=req.payee,
        inflow=req.inflow,
        outflow=req.outflow,
        category=res.category,
    )
    for req, res in zip(txns, results)
]

for item in complete:
    print(item)
