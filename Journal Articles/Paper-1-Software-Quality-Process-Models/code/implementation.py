"""
Implementation of Software Quality Process Models Evaluation Framework
Authors: Dr. Tanzila Kehkashan et al.
Paper: Sw Quality Process Models: An Appraisal
DOI: 10.4172/2165-7866.1000202

This implementation provides tools for theoretical and empirical evaluation
of software quality models including ISO 9001, SPICE, and SW-CMM.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class QualityModel:
    """Base class for software quality models"""

    def __init__(self, name: str):
        self.name = name
        self.requirements = {}
        self.evaluation_results = {}

    def add_requirement(self, req_id: str, description: str, category: str):
        """Add a quality requirement to the model"""
        self.requirements[req_id] = {
            'description': description,
            'category': category,
            'status': 'Not Evaluated'
        }

    def evaluate_requirement(self, req_id: str, status: str):
        """Evaluate a requirement: Satisfied, Partially Satisfied, Not Satisfied"""
        if req_id in self.requirements:
            self.requirements[req_id]['status'] = status
            self.evaluation_results[req_id] = status


class ISO9001Model(QualityModel):
    """ISO 9001:1994 Quality Model Implementation"""

    def __init__(self):
        super().__init__("ISO 9001:1994")
        self._initialize_requirements()

    def _initialize_requirements(self):
        """Initialize ISO 9001 requirements"""
        categories = [
            "Management Responsibility",
            "Quality System",
            "Contract Review",
            "Design Control",
            "Document and Data Control",
            "Purchasing",
            "Product Identification",
            "Process Control",
            "Inspection and Testing",
            "Quality Records",
            "Quality Audits",
            "Training"
        ]

        for i, category in enumerate(categories, 1):
            self.add_requirement(f"ISO_{i}", f"{category} compliance", category)


class SPICEModel(QualityModel):
    """SPICE v1.3 Process Assessment Model"""

    def __init__(self):
        super().__init__("SPICE v1.3")
        self._initialize_capabilities()

    def _initialize_capabilities(self):
        """Initialize SPICE capability levels"""
        capabilities = [
            "Process Performance",
            "Performance Management",
            "Work Product Management",
            "Process Definition",
            "Process Deployment",
            "Process Measurement",
            "Process Control",
            "Process Innovation",
            "Process Optimization"
        ]

        for i, cap in enumerate(capabilities, 1):
            self.add_requirement(f"SPICE_{i}", cap, "Capability Assessment")


class SWCMMModel(QualityModel):
    """SW-CMM v1.3 Capability Maturity Model"""

    def __init__(self):
        super().__init__("SW-CMM v1.3")
        self._initialize_maturity_levels()

    def _initialize_maturity_levels(self):
        """Initialize CMM maturity levels and key process areas"""
        kpas = {
            "Level 2": ["Requirements Management", "Project Planning",
                       "Project Tracking", "Quality Assurance",
                       "Configuration Management"],
            "Level 3": ["Organization Process Focus", "Organization Process Definition",
                       "Training Program", "Integrated Software Management",
                       "Peer Reviews"],
            "Level 4": ["Quantitative Process Management", "Software Quality Management"],
            "Level 5": ["Defect Prevention", "Technology Change Management",
                       "Process Change Management"]
        }

        idx = 1
        for level, areas in kpas.items():
            for area in areas:
                self.add_requirement(f"CMM_{idx}", area, level)
                idx += 1


class QualityModelEvaluator:
    """Framework for evaluating software quality models"""

    def __init__(self):
        self.models = []
        self.case_studies = []
        self.results = {}

    def add_model(self, model: QualityModel):
        """Add a quality model for evaluation"""
        self.models.append(model)

    def add_case_study(self, name: str, characteristics: Dict):
        """Add a case study for empirical evaluation"""
        self.case_studies.append({
            'name': name,
            'characteristics': characteristics
        })

    def theoretical_evaluation(self, model: QualityModel,
                             user_requirements: Dict[str, List[str]]):
        """
        Perform theoretical evaluation of a quality model

        Args:
            model: Quality model to evaluate
            user_requirements: Dictionary mapping user classes to requirements
        """
        print(f"\n{'='*60}")
        print(f"Theoretical Evaluation: {model.name}")
        print(f"{'='*60}\n")

        total_requirements = len(model.requirements)
        satisfied_count = 0
        partial_count = 0
        not_satisfied_count = 0

        for req_id, req_data in model.requirements.items():
            status = req_data['status']
            if status == 'Satisfied':
                satisfied_count += 1
            elif status == 'Partially Satisfied':
                partial_count += 1
            elif status == 'Not Satisfied':
                not_satisfied_count += 1

        # Calculate percentages
        satisfied_pct = (satisfied_count / total_requirements) * 100
        partial_pct = (partial_count / total_requirements) * 100
        not_satisfied_pct = (not_satisfied_count / total_requirements) * 100

        print(f"Total Requirements: {total_requirements}")
        print(f"Satisfied: {satisfied_count} ({satisfied_pct:.2f}%)")
        print(f"Partially Satisfied: {partial_count} ({partial_pct:.2f}%)")
        print(f"Not Satisfied: {not_satisfied_count} ({not_satisfied_pct:.2f}%)")

        return {
            'total': total_requirements,
            'satisfied': satisfied_count,
            'partial': partial_count,
            'not_satisfied': not_satisfied_count,
            'satisfied_pct': satisfied_pct
        }

    def empirical_evaluation(self, model: QualityModel, case_study: Dict):
        """Perform empirical evaluation using case study data"""
        print(f"\n{'='*60}")
        print(f"Empirical Evaluation: {model.name}")
        print(f"Case Study: {case_study['name']}")
        print(f"{'='*60}\n")

        # Simulate evaluation based on case study characteristics
        characteristics = case_study['characteristics']

        evaluation_score = 0
        max_score = len(characteristics)

        for char, value in characteristics.items():
            print(f"{char}: {value}")
            if value in ['Yes', 'High', 'Good']:
                evaluation_score += 1
            elif value in ['Partial', 'Medium']:
                evaluation_score += 0.5

        performance = (evaluation_score / max_score) * 100
        print(f"\nOverall Performance: {performance:.2f}%")

        return performance

    def comparative_analysis(self):
        """Compare performance across all models"""
        print(f"\n{'='*60}")
        print("Comparative Analysis of Quality Models")
        print(f"{'='*60}\n")

        comparison_data = []

        for model in self.models:
            model_results = self.theoretical_evaluation(model, {})
            comparison_data.append({
                'Model': model.name,
                'Satisfied (%)': model_results['satisfied_pct'],
                'Total Requirements': model_results['total']
            })

        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        return df

    def visualize_results(self, results_df: pd.DataFrame):
        """Visualize model comparison results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot of satisfaction percentages
        ax1.bar(results_df['Model'], results_df['Satisfied (%)'],
                color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.set_ylabel('Satisfaction Percentage (%)')
        ax1.set_title('Model Satisfaction Comparison')
        ax1.set_ylim(0, 100)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Requirements count comparison
        ax2.bar(results_df['Model'], results_df['Total Requirements'],
                color=['#9b59b6', '#f39c12', '#1abc9c'])
        ax2.set_ylabel('Number of Requirements')
        ax2.set_title('Total Requirements per Model')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('quality_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'quality_model_comparison.png'")
        plt.show()


def main():
    """Main execution function demonstrating the framework"""

    print("="*60)
    print("Software Quality Process Models Evaluation Framework")
    print("="*60)

    # Initialize evaluator
    evaluator = QualityModelEvaluator()

    # Create quality models
    iso_model = ISO9001Model()
    spice_model = SPICEModel()
    cmm_model = SWCMMModel()

    # Simulate some evaluations (in real scenario, this would be based on actual assessment)
    # For ISO model
    for i in range(1, 6):
        iso_model.evaluate_requirement(f"ISO_{i}", "Satisfied")
    for i in range(6, 9):
        iso_model.evaluate_requirement(f"ISO_{i}", "Partially Satisfied")
    for i in range(9, 13):
        iso_model.evaluate_requirement(f"ISO_{i}", "Not Satisfied")

    # For SPICE model
    for i in range(1, 5):
        spice_model.evaluate_requirement(f"SPICE_{i}", "Satisfied")
    for i in range(5, 7):
        spice_model.evaluate_requirement(f"SPICE_{i}", "Partially Satisfied")
    for i in range(7, 10):
        spice_model.evaluate_requirement(f"SPICE_{i}", "Not Satisfied")

    # For CMM model
    for i in range(1, 8):
        cmm_model.evaluate_requirement(f"CMM_{i}", "Satisfied")
    for i in range(8, 12):
        cmm_model.evaluate_requirement(f"CMM_{i}", "Partially Satisfied")
    for i in range(12, 18):
        cmm_model.evaluate_requirement(f"CMM_{i}", "Not Satisfied")

    # Add models to evaluator
    evaluator.add_model(iso_model)
    evaluator.add_model(spice_model)
    evaluator.add_model(cmm_model)

    # Add case studies
    evaluator.add_case_study("University of Durham", {
        'Well-defined process': 'Yes',
        'Documentation': 'Good',
        'Team experience': 'Medium',
        'Tool support': 'Partial',
        'Quality focus': 'Low'
    })

    evaluator.add_case_study("GNU GCC Project", {
        'Well-defined process': 'Yes',
        'Documentation': 'High',
        'Team experience': 'High',
        'Tool support': 'Good',
        'Quality focus': 'High'
    })

    # Perform comparative analysis
    results_df = evaluator.comparative_analysis()

    # Perform empirical evaluations
    for case_study in evaluator.case_studies:
        for model in evaluator.models:
            evaluator.empirical_evaluation(model, case_study)

    # Visualize results
    evaluator.visualize_results(results_df)

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
