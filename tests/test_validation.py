"""
tests/test_validation.py â€” Unit tests for data validation business rules.
These run on every push via GitHub Actions CI.
"""
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline.validate import run_bank_business_rules, run_cfpb_business_rules
@pytest.fixture
def valid_bank_df():
    return pd.DataFrame({
        "age": [35, 42, 28],
        "job": ["admin.", "technician", "services"],
        "marital": ["married", "single", "divorced"],
        "education": ["secondary", "tertiary", "primary"],
        "default": ["no", "no", "yes"],
        "balance": [1500, 200, -100],
        "housing": ["yes", "no", "yes"],
        "loan": ["no", "no", "yes"],
        "contact": ["cellular", "telephone", "cellular"],
        "day": [5, 12, 20],
        "month": ["may", "jun", "jul"],
        "duration": [180, 300, 120],
        "campaign": [1, 2, 3],
        "pdays": [-1, 180, -1],
        "previous": [0, 1, 0],
        "poutcome": ["unknown", "success", "unknown"],
        "y": ["no", "yes", "no"],
    })
@pytest.fixture
def valid_cfpb_df():
    return pd.DataFrame({
        "Complaint ID": ["101", "102", "103", "104", "105", "106"],
        "Product": [
            "Mortgage",
            "Credit card or prepaid card",
            "Checking or savings account",
            "Student loan",
            "Debt collection",
            "Vehicle loan or lease",
        ],
        "Consumer complaint narrative": [
            "My bank charged me extra fees without any prior notice and I was completely unable to obtain a refund after multiple attempts to contact customer service.",
            "I disputed a fraudulent charge on my credit card statement and the bank never responded to my formal dispute request within the required time window.",
            "My checking account was closed without any warning and I lost access to my funds for over two weeks causing significant financial hardship.",
            "My student loan servicer applied my payment incorrectly and despite calling several times the issue remained unresolved for months.",
            "A debt collector contacted me repeatedly about a debt that had already been paid and settled more than two years ago.",
            "The dealer added unauthorized add-on fees to my vehicle loan contract that I did not agree to during the purchase negotiation.",
        ],
        "Company": ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E", "Bank F"],
    })
class TestBankBusinessRules:
    def test_all_rules_pass_on_valid_data(self, valid_bank_df):
        errors = run_bank_business_rules(valid_bank_df)
        assert errors == [], f"Unexpected errors: {errors}"
    def test_biz01_missing_target_column(self, valid_bank_df):
        df = valid_bank_df.drop(columns=["y"])
        errors = run_bank_business_rules(df)
        assert any("BIZ-01" in e for e in errors)
    def test_biz02_minority_class_too_low(self, valid_bank_df):
        df = valid_bank_df.copy()
        df["y"] = "no"
        errors = run_bank_business_rules(df)
        assert any("BIZ-02" in e for e in errors)
    def test_biz04_excessive_campaign_contacts(self, valid_bank_df):
        df = valid_bank_df.copy()
        df.loc[0, "campaign"] = 99
        errors = run_bank_business_rules(df)
        assert any("BIZ-04" in e for e in errors)
    def test_biz05_zero_duration_with_yes(self, valid_bank_df):
        df = valid_bank_df.copy()
        df.loc[0, "duration"] = 0
        df.loc[0, "y"] = "yes"
        errors = run_bank_business_rules(df)
        assert any("BIZ-05" in e for e in errors)
class TestCFPBBusinessRules:
    def test_all_rules_pass_on_valid_data(self, valid_cfpb_df):
        errors = run_cfpb_business_rules(valid_cfpb_df)
        assert errors == [], f"Unexpected errors: {errors}"
    def test_cfpb02_duplicate_complaint_ids(self, valid_cfpb_df):
        df = valid_cfpb_df.copy()
        df.loc[1, "Complaint ID"] = "101"
        errors = run_cfpb_business_rules(df)
        assert any("CFPB-02" in e for e in errors)
    def test_cfpb05_too_few_products(self, valid_cfpb_df):
        df = valid_cfpb_df.copy()
        df["Product"] = "Mortgage"
        errors = run_cfpb_business_rules(df)
        assert any("CFPB-05" in e for e in errors)
