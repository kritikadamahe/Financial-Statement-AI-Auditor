import json

new_data = {
    "Satyam": {
        "label": 1,
        "industry": "IT Services",
        "fraud_type": "Fabricated $1B in cash balances, inflated revenues",
        "nlp_risk_score": 85,
        "currency": "INR Crores",
        "data": {
            "2004": {"Revenue": 2542, "Cost of Goods Sold": 1480, "Gross Profit": 1062, "Operating Expenses": 371, "Net Income": 556, "Current Assets": 1500, "Total Assets": 2588, "Current Liabilities": 86, "Total Debt": 2, "Total Equity": 2581, "Operating Cash Flow": 400},
            "2005": {"Revenue": 3464, "Cost of Goods Sold": 2125, "Gross Profit": 1339, "Operating Expenses": 416, "Net Income": 750, "Current Assets": 2000, "Total Assets": 3227, "Current Liabilities": 139, "Total Debt": 18, "Total Equity": 3088, "Operating Cash Flow": 600},
            "2006": {"Revenue": 4634, "Cost of Goods Sold": 2865, "Gross Profit": 1769, "Operating Expenses": 545, "Net Income": 1240, "Current Assets": 2800, "Total Assets": 4348, "Current Liabilities": 211, "Total Debt": 22, "Total Equity": 4137, "Operating Cash Flow": 800},
            "2007": {"Revenue": 6228, "Cost of Goods Sold": 3923, "Gross Profit": 2305, "Operating Expenses": 765, "Net Income": 1423, "Current Assets": 3800, "Total Assets": 5803, "Current Liabilities": 333, "Total Debt": 25, "Total Equity": 5470, "Operating Cash Flow": 1100},
            "2008": {"Revenue": 8473, "Cost of Goods Sold": 5340, "Gross Profit": 3133, "Operating Expenses": 1062, "Net Income": 1716, "Current Assets": 5000, "Total Assets": 7381, "Current Liabilities": 400, "Total Debt": 30, "Total Equity": 7358, "Operating Cash Flow": 1413}
        }
    },
    "Toshiba": {
        "label": 1,
        "industry": "Conglomerate",
        "fraud_type": "Overstated profits by $1.2B via percentage-of-completion manipulation",
        "nlp_risk_score": 75,
        "currency": "JPY Billions",
        "data": {
            "2010": {"Revenue": 6130, "Cost of Goods Sold": 4711, "Gross Profit": 1419, "Operating Expenses": 1312, "Net Income": -198, "Current Assets": 3147, "Total Assets": 5994, "Current Liabilities": 2488, "Total Debt": 1218, "Total Equity": 797, "Operating Cash Flow": 451},
            "2011": {"Revenue": 6271, "Cost of Goods Sold": 4782, "Gross Profit": 1489, "Operating Expenses": 1293, "Net Income": 118, "Current Assets": 3096, "Total Assets": 6056, "Current Liabilities": 3068, "Total Debt": 1230, "Total Equity": 896, "Operating Cash Flow": 54},
            "2012": {"Revenue": 5994, "Cost of Goods Sold": 4539, "Gross Profit": 1456, "Operating Expenses": 1251, "Net Income": 78, "Current Assets": 3037, "Total Assets": 6171, "Current Liabilities": 2738, "Total Debt": 1489, "Total Equity": 948, "Operating Cash Flow": -148},
            "2013": {"Revenue": 5727, "Cost of Goods Sold": 4314, "Gross Profit": 1413, "Operating Expenses": 1122, "Net Income": 77, "Current Assets": 3247, "Total Assets": 5857, "Current Liabilities": 2734, "Total Debt": 1425, "Total Equity": 1030, "Operating Cash Flow": -190},
            "2014": {"Revenue": 6503, "Cost of Goods Sold": 4854, "Gross Profit": 1648, "Operating Expenses": 1357, "Net Income": 51, "Current Assets": 3363, "Total Assets": 6243, "Current Liabilities": 2869, "Total Debt": 1358, "Total Equity": 1185, "Operating Cash Flow": -243}
        }
    },
    "Tyco": {
        "label": 1,
        "industry": "Conglomerate",
        "fraud_type": "Aggressive acquisition accounting, CEO looting",
        "nlp_risk_score": 88,
        "currency": "USD Millions",
        "data": {
            "1999": {"Revenue": 22494, "Cost of Goods Sold": 15750, "Gross Profit": 6744, "Operating Expenses": 5350, "Net Income": 874, "Current Assets": 11500, "Total Assets": 32362, "Current Liabilities": 8500, "Total Debt": 12000, "Total Equity": 12333, "Operating Cash Flow": 2800},
            "2000": {"Revenue": 28928, "Cost of Goods Sold": 20250, "Gross Profit": 8677, "Operating Expenses": 3580, "Net Income": 4319, "Current Assets": 14500, "Total Assets": 40404, "Current Liabilities": 10500, "Total Debt": 14000, "Total Equity": 17033, "Operating Cash Flow": 3200},
            "2001": {"Revenue": 34002, "Cost of Goods Sold": 23800, "Gross Profit": 10202, "Operating Expenses": 5230, "Net Income": 3464, "Current Assets": 42000, "Total Assets": 111287, "Current Liabilities": 35000, "Total Debt": 30000, "Total Equity": 31997, "Operating Cash Flow": 3500},
            "2002": {"Revenue": 35590, "Cost of Goods Sold": 25100, "Gross Profit": 10490, "Operating Expenses": 12550, "Net Income": -9180, "Current Assets": 19639, "Total Assets": 65458, "Current Liabilities": 17524, "Total Debt": 24248, "Total Equity": 23000, "Operating Cash Flow": 779},
            "2003": {"Revenue": 36801, "Cost of Goods Sold": 25860, "Gross Profit": 10941, "Operating Expenses": 9350, "Net Income": 980, "Current Assets": 17240, "Total Assets": 63545, "Current Liabilities": 10572, "Total Debt": 20969, "Total Equity": 26000, "Operating Cash Flow": 3200}
        }
    },
    "HealthSouth": {
        "label": 1,
        "industry": "Healthcare",
        "fraud_type": "Fictitious assets, capitalized operating expenses",
        "nlp_risk_score": 78,
        "currency": "USD Millions",
        "data": {
            "1998": {"Revenue": 2900, "Cost of Goods Sold": 2400, "Gross Profit": 500, "Operating Expenses": 350, "Net Income": 47, "Current Assets": 1200, "Total Assets": 6778, "Current Liabilities": 800, "Total Debt": 3300, "Total Equity": 2500, "Operating Cash Flow": 350},
            "1999": {"Revenue": 3000, "Cost of Goods Sold": 2500, "Gross Profit": 500, "Operating Expenses": 350, "Net Income": 77, "Current Assets": 1200, "Total Assets": 6891, "Current Liabilities": 850, "Total Debt": 3500, "Total Equity": 2500, "Operating Cash Flow": 350},
            "2000": {"Revenue": 3300, "Cost of Goods Sold": 2600, "Gross Profit": 700, "Operating Expenses": 300, "Net Income": 279, "Current Assets": 1300, "Total Assets": 7380, "Current Liabilities": 900, "Total Debt": 3600, "Total Equity": 2700, "Operating Cash Flow": 350},
            "2001": {"Revenue": 4370, "Cost of Goods Sold": 3500, "Gross Profit": 870, "Operating Expenses": 550, "Net Income": 202, "Current Assets": 1400, "Total Assets": 7579, "Current Liabilities": 950, "Total Debt": 3800, "Total Equity": 2800, "Operating Cash Flow": 350},
            "2002": {"Revenue": 3600, "Cost of Goods Sold": 3000, "Gross Profit": 600, "Operating Expenses": 400, "Net Income": 136, "Current Assets": 1400, "Total Assets": 7931, "Current Liabilities": 1000, "Total Debt": 4000, "Total Equity": 2900, "Operating Cash Flow": 350}
        }
    },
    "Waste Management": {
        "label": 1,
        "industry": "Waste Management",
        "fraud_type": "Improper depreciation and capitalization",
        "nlp_risk_score": 72,
        "currency": "USD Millions",
        "data": {
            "1993": {"Revenue": 8480, "Cost of Goods Sold": 6500, "Gross Profit": 1980, "Operating Expenses": 1000, "Net Income": 453, "Current Assets": 3000, "Total Assets": 15000, "Current Liabilities": 3000, "Total Debt": 5500, "Total Equity": 5500, "Operating Cash Flow": 1750},
            "1994": {"Revenue": 9050, "Cost of Goods Sold": 6900, "Gross Profit": 2150, "Operating Expenses": 1000, "Net Income": 784, "Current Assets": 3200, "Total Assets": 16000, "Current Liabilities": 3200, "Total Debt": 6000, "Total Equity": 6000, "Operating Cash Flow": 1750},
            "1995": {"Revenue": 9190, "Cost of Goods Sold": 7100, "Gross Profit": 2090, "Operating Expenses": 1050, "Net Income": 604, "Current Assets": 3300, "Total Assets": 17000, "Current Liabilities": 3300, "Total Debt": 6500, "Total Equity": 6300, "Operating Cash Flow": 1750},
            "1996": {"Revenue": 9600, "Cost of Goods Sold": 7300, "Gross Profit": 2300, "Operating Expenses": 1100, "Net Income": 192, "Current Assets": 3400, "Total Assets": 18400, "Current Liabilities": 3500, "Total Debt": 7000, "Total Equity": 6000, "Operating Cash Flow": 1750},
            "1997": {"Revenue": 11970, "Cost of Goods Sold": 9500, "Gross Profit": 2470, "Operating Expenses": 1300, "Net Income": 300, "Current Assets": 3500, "Total Assets": 20000, "Current Liabilities": 4000, "Total Debt": 8000, "Total Equity": 4000, "Operating Cash Flow": 1750}
        }
    },
    "Apple": {
        "label": 0,
        "industry": "Technology",
        "fraud_type": None,
        "nlp_risk_score": 5,
        "currency": "USD Millions",
        "data": {
            "2017": {"Revenue": 229234, "Cost of Goods Sold": 141048, "Gross Profit": 88186, "Operating Expenses": 26852, "Net Income": 48351, "Current Assets": 128645, "Total Assets": 375319, "Current Liabilities": 100814, "Total Debt": 115663, "Total Equity": 134047, "Operating Cash Flow": 63598},
            "2018": {"Revenue": 265595, "Cost of Goods Sold": 163756, "Gross Profit": 101839, "Operating Expenses": 30941, "Net Income": 59531, "Current Assets": 131339, "Total Assets": 365725, "Current Liabilities": 115929, "Total Debt": 114488, "Total Equity": 107147, "Operating Cash Flow": 77434},
            "2019": {"Revenue": 260174, "Cost of Goods Sold": 161782, "Gross Profit": 98392, "Operating Expenses": 34622, "Net Income": 55256, "Current Assets": 162819, "Total Assets": 338516, "Current Liabilities": 105718, "Total Debt": 108047, "Total Equity": 90488, "Operating Cash Flow": 69391},
            "2020": {"Revenue": 274515, "Cost of Goods Sold": 169559, "Gross Profit": 104956, "Operating Expenses": 38668, "Net Income": 57411, "Current Assets": 143713, "Total Assets": 323888, "Current Liabilities": 105392, "Total Debt": 112436, "Total Equity": 65339, "Operating Cash Flow": 80674},
            "2021": {"Revenue": 365817, "Cost of Goods Sold": 212981, "Gross Profit": 152836, "Operating Expenses": 43887, "Net Income": 94680, "Current Assets": 134836, "Total Assets": 351002, "Current Liabilities": 125481, "Total Debt": 124719, "Total Equity": 63090, "Operating Cash Flow": 104038}
        }
    },
    "Google": {
        "label": 0,
        "industry": "Technology",
        "fraud_type": None,
        "nlp_risk_score": 6,
        "currency": "USD Millions",
        "data": {
            "2017": {"Revenue": 110855, "Cost of Goods Sold": 45583, "Gross Profit": 65272, "Operating Expenses": 39102, "Net Income": 12662, "Current Assets": 127624, "Total Assets": 197295, "Current Liabilities": 32875, "Total Debt": 3960, "Total Equity": 152502, "Operating Cash Flow": 37133},
            "2018": {"Revenue": 136819, "Cost of Goods Sold": 59549, "Gross Profit": 77270, "Operating Expenses": 49746, "Net Income": 30736, "Current Assets": 136543, "Total Assets": 232792, "Current Liabilities": 37908, "Total Debt": 4010, "Total Equity": 177628, "Operating Cash Flow": 44484},
            "2019": {"Revenue": 161857, "Cost of Goods Sold": 71896, "Gross Profit": 89961, "Operating Expenses": 55671, "Net Income": 34343, "Current Assets": 163892, "Total Assets": 275909, "Current Liabilities": 42958, "Total Debt": 15960, "Total Equity": 201442, "Operating Cash Flow": 54588},
            "2020": {"Revenue": 182527, "Cost of Goods Sold": 84732, "Gross Profit": 97795, "Operating Expenses": 56577, "Net Income": 40269, "Current Assets": 200674, "Total Assets": 319616, "Current Liabilities": 49236, "Total Debt": 26770, "Total Equity": 222544, "Operating Cash Flow": 65124},
            "2021": {"Revenue": 257637, "Cost of Goods Sold": 110939, "Gross Profit": 146698, "Operating Expenses": 67938, "Net Income": 76033, "Current Assets": 230192, "Total Assets": 359268, "Current Liabilities": 55906, "Total Debt": 28390, "Total Equity": 251635, "Operating Cash Flow": 91652}
        }
    },
    "Walmart": {
        "label": 0,
        "industry": "Retail",
        "fraud_type": None,
        "nlp_risk_score": 8,
        "currency": "USD Millions",
        "data": {
            "2017": {"Revenue": 485873, "Cost of Goods Sold": 361256, "Gross Profit": 124617, "Operating Expenses": 103046, "Net Income": 13643, "Current Assets": 67735, "Total Assets": 198825, "Current Liabilities": 78740, "Total Debt": 51770, "Total Equity": 81417, "Operating Cash Flow": 31529},
            "2018": {"Revenue": 500343, "Cost of Goods Sold": 373397, "Gross Profit": 126946, "Operating Expenses": 106510, "Net Income": 9862, "Current Assets": 70894, "Total Assets": 204522, "Current Liabilities": 81550, "Total Debt": 47941, "Total Equity": 77880, "Operating Cash Flow": 28337},
            "2019": {"Revenue": 514405, "Cost of Goods Sold": 385301, "Gross Profit": 129104, "Operating Expenses": 107147, "Net Income": 6670, "Current Assets": 73382, "Total Assets": 219295, "Current Liabilities": 87466, "Total Debt": 52668, "Total Equity": 79836, "Operating Cash Flow": 27753},
            "2020": {"Revenue": 523964, "Cost of Goods Sold": 394605, "Gross Profit": 129359, "Operating Expenses": 108791, "Net Income": 14881, "Current Assets": 77163, "Total Assets": 236495, "Current Liabilities": 96066, "Total Debt": 54642, "Total Equity": 81552, "Operating Cash Flow": 25255},
            "2021": {"Revenue": 559151, "Cost of Goods Sold": 420315, "Gross Profit": 138836, "Operating Expenses": 116288, "Net Income": 13510, "Current Assets": 82324, "Total Assets": 252496, "Current Liabilities": 102920, "Total Debt": 54342, "Total Equity": 90341, "Operating Cash Flow": 36074}
        }
    },
    "Coca-Cola": {
        "label": 0,
        "industry": "Consumer Goods",
        "fraud_type": None,
        "nlp_risk_score": 7,
        "currency": "USD Millions",
        "data": {
            "2017": {"Revenue": 36212, "Cost of Goods Sold": 13256, "Gross Profit": 22956, "Operating Expenses": 12367, "Net Income": 1248, "Current Assets": 36151, "Total Assets": 87896, "Current Liabilities": 27243, "Total Debt": 44585, "Total Equity": 25764, "Operating Cash Flow": 6971},
            "2018": {"Revenue": 34300, "Cost of Goods Sold": 13858, "Gross Profit": 20442, "Operating Expenses": 13640, "Net Income": 6434, "Current Assets": 25603, "Total Assets": 83216, "Current Liabilities": 25625, "Total Debt": 43158, "Total Equity": 18977, "Operating Cash Flow": 6997},
            "2019": {"Revenue": 37266, "Cost of Goods Sold": 14619, "Gross Profit": 22647, "Operating Expenses": 14565, "Net Income": 8920, "Current Assets": 20411, "Total Assets": 86381, "Current Liabilities": 20411, "Total Debt": 42610, "Total Equity": 21098, "Operating Cash Flow": 10471},
            "2020": {"Revenue": 33014, "Cost of Goods Sold": 13433, "Gross Profit": 19581, "Operating Expenses": 13101, "Net Income": 7747, "Current Assets": 19240, "Total Assets": 87296, "Current Liabilities": 25723, "Total Debt": 45951, "Total Equity": 21284, "Operating Cash Flow": 9781},
            "2021": {"Revenue": 38655, "Cost of Goods Sold": 15357, "Gross Profit": 23298, "Operating Expenses": 14484, "Net Income": 9771, "Current Assets": 22545, "Total Assets": 94354, "Current Liabilities": 22238, "Total Debt": 42933, "Total Equity": 24860, "Operating Cash Flow": 12625}
        }
    },
    "IBM": {
        "label": 0,
        "industry": "Technology",
        "fraud_type": None,
        "nlp_risk_score": 11,
        "currency": "USD Millions",
        "data": {
            "2017": {"Revenue": 79139, "Cost of Goods Sold": 42196, "Gross Profit": 36943, "Operating Expenses": 25270, "Net Income": 5753, "Current Assets": 44524, "Total Assets": 125356, "Current Liabilities": 32836, "Total Debt": 46812, "Total Equity": 17725, "Operating Cash Flow": 16881},
            "2018": {"Revenue": 79591, "Cost of Goods Sold": 42655, "Gross Profit": 36936, "Operating Expenses": 24745, "Net Income": 8728, "Current Assets": 49146, "Total Assets": 123382, "Current Liabilities": 36836, "Total Debt": 46677, "Total Equity": 16795, "Operating Cash Flow": 15945},
            "2019": {"Revenue": 77147, "Cost of Goods Sold": 40659, "Gross Profit": 36488, "Operating Expenses": 24634, "Net Income": 9431, "Current Assets": 38421, "Total Assets": 152186, "Current Liabilities": 34057, "Total Debt": 62945, "Total Equity": 20841, "Operating Cash Flow": 14800},
            "2020": {"Revenue": 55179, "Cost of Goods Sold": 24314, "Gross Profit": 30865, "Operating Expenses": 26823, "Net Income": 5590, "Current Assets": 39165, "Total Assets": 155971, "Current Liabilities": 34923, "Total Debt": 61489, "Total Equity": 20598, "Operating Cash Flow": 18200},
            "2021": {"Revenue": 57351, "Cost of Goods Sold": 25865, "Gross Profit": 31486, "Operating Expenses": 25233, "Net Income": 5743, "Current Assets": 29540, "Total Assets": 132001, "Current Liabilities": 31106, "Total Debt": 51704, "Total Equity": 18901, "Operating Cash Flow": 12800}
        }
    },
    "Intel": {
        "label": 0,
        "industry": "Semiconductors",
        "fraud_type": None,
        "nlp_risk_score": 9,
        "currency": "USD Millions",
        "data": {
            "2017": {"Revenue": 62761, "Cost of Goods Sold": 23663, "Gross Profit": 39098, "Operating Expenses": 21048, "Net Income": 9601, "Current Assets": 47249, "Total Assets": 123249, "Current Liabilities": 24754, "Total Debt": 29001, "Total Equity": 69019, "Operating Cash Flow": 22190},
            "2018": {"Revenue": 70848, "Cost of Goods Sold": 27111, "Gross Profit": 43737, "Operating Expenses": 20421, "Net Income": 21053, "Current Assets": 57718, "Total Assets": 127998, "Current Liabilities": 27462, "Total Debt": 36401, "Total Equity": 70305, "Operating Cash Flow": 23280},
            "2019": {"Revenue": 71965, "Cost of Goods Sold": 29825, "Gross Profit": 42140, "Operating Expenses": 20105, "Net Income": 21048, "Current Assets": 58401, "Total Assets": 136522, "Current Liabilities": 29322, "Total Debt": 38101, "Total Equity": 81399, "Operating Cash Flow": 33151},
            "2020": {"Revenue": 77867, "Cost of Goods Sold": 34255, "Gross Profit": 43612, "Operating Expenses": 19934, "Net Income": 20899, "Current Assets": 75548, "Total Assets": 153091, "Current Liabilities": 31438, "Total Debt": 39523, "Total Equity": 96081, "Operating Cash Flow": 35385},
            "2021": {"Revenue": 79024, "Cost of Goods Sold": 35209, "Gross Profit": 43815, "Operating Expenses": 24359, "Net Income": 19868, "Current Assets": 66289, "Total Assets": 168406, "Current Liabilities": 27462, "Total Debt": 33510, "Total Equity": 106793, "Operating Cash Flow": 30064}
        }
    }
}

with open(r'c:\Users\Admin\Desktop\FinAuditAI\Financial-Statement-AI-Auditor\backend\real_world_data.json', 'r') as f:
    existing_data = json.load(f)

existing_data.update(new_data)

with open(r'c:\Users\Admin\Desktop\FinAuditAI\Financial-Statement-AI-Auditor\backend\real_world_data.json', 'w') as f:
    json.dump(existing_data, f, indent=2)

print("Updated real_world_data.json with", len(new_data), "new companies.")
