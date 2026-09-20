import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "wan_t2v.py"
SPEC = importlib.util.spec_from_file_location("wan_t2v", SCRIPT)
wan_t2v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wan_t2v)


class WanWorkflowTest(unittest.TestCase):
    def test_previous_chinese_prompt_builds_official_wan_workflow(self):
        workflow = wan_t2v.build_workflow(wan_t2v.DEFAULT_PROMPT, frames=17, steps=12)

        self.assertEqual(workflow["4"]["inputs"]["text"], wan_t2v.DEFAULT_PROMPT)
        self.assertEqual(workflow["2"]["inputs"]["type"], "wan")
        self.assertEqual(workflow["2"]["inputs"]["clip_name"], "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
        self.assertEqual(workflow["3"]["inputs"]["vae_name"], "wan_2.1_vae.safetensors")
        self.assertEqual(workflow["6"]["inputs"]["length"], 17)
        self.assertEqual(workflow["8"]["inputs"]["steps"], 12)
        self.assertEqual(workflow["8"]["inputs"]["model"], ["1", 0])


if __name__ == "__main__":
    unittest.main()
