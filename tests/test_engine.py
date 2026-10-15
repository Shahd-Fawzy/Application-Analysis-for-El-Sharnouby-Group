import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine

class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sales=engine.sales()
        cls.fc,cls.val,cls.wape=engine.forecast_and_validation(cls.sales)
    def test_sales_data(self):
        self.assertGreater(len(self.sales),1000)
        self.assertGreater(self.sales.Net_Sales_EGP.sum(),0)
    def test_forecast_no_negative(self):
        self.assertTrue((self.fc.Forecast_30D_Units>=0).all())
        self.assertGreaterEqual(self.wape,0)
        self.assertLess(self.val.Month.max(),self.sales.Month.max()+__import__('pandas').offsets.MonthBegin(1))
    def test_inventory_replenishment(self):
        v=engine.inventory_actions(self.fc)
        self.assertGreater(len(v),0)
        self.assertTrue((v.Suggested_Order_Units>=0).all())
        self.assertTrue((v.Potential_Lead_Time_Shortfall_Units>=0).all())
    def test_scenario_neutral(self):
        v=engine.scenario(self.sales)
        self.assertAlmostEqual(v['Base_Revenue_EGP'],v['Scenario_Revenue_EGP'],places=5)
        self.assertAlmostEqual(v['Indicative_Profit_Change_After_Incremental_Marketing_EGP'],0,places=4)
    def test_other_modules(self):
        customers=engine.rfm_customers(self.sales)
        self.assertTrue(customers.Orders.min()>=1)
        self.assertFalse(engine.cohorts(self.sales).empty)
        lines,machines,quality=engine.factory_and_printing()
        self.assertFalse(lines.empty or machines.empty or quality.empty)
        self.assertFalse(engine.suppliers().empty)
        self.assertFalse(engine.marketing().empty)
        actions=engine.action_register(engine.inventory_actions(self.fc),machines,lines,engine.suppliers(),customers)
        self.assertGreater(len(actions),0)

if __name__=='__main__':unittest.main()
