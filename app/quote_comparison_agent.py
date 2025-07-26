# quote_comparison_agent.py
from autogen import ConversableAgent
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import json
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
from enum import Enum
import pandas as pd
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class ComparisonMetric(Enum):
    PRICE = "price"
    DELIVERY_TIME = "delivery_time"
    COMPLIANCE = "compliance"
    VENDOR_RATING = "vendor_rating"
    TOTAL_COST = "total_cost"

class QuoteStatus(Enum):
    PENDING = "pending"
    RECEIVED = "received"
    ANALYZED = "analyzed"
    SELECTED = "selected"
    REJECTED = "rejected"

@dataclass
class LineItemQuote:
    part_number: str
    description: str
    quantity: int
    unit_price: float
    total_price: float
    currency: str
    lead_time_days: Optional[int] = None
    compliance_score: float = 0.0
    notes: Optional[str] = None

@dataclass
class VendorQuote:
    quote_id: str
    vendor_name: str
    vendor_email: str
    rfq_id: str
    submission_date: datetime
    line_items: List[LineItemQuote]
    total_quote_value: float
    currency: str
    delivery_terms: Optional[str] = None
    payment_terms: Optional[str] = None
    validity_period_days: int = 30
    compliance_documents: List[str] = None
    status: QuoteStatus = QuoteStatus.RECEIVED
    vendor_rating: float = 0.0
    shipping_cost: float = 0.0
    tax_amount: float = 0.0
    discount_percentage: float = 0.0
    
    def __post_init__(self):
        if self.compliance_documents is None:
            self.compliance_documents = []

class QuoteComparison(BaseModel):
    rfq_id: str
    rfq_title: str
    comparison_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    vendor_quotes: List[Dict[str, Any]] = []
    comparison_matrix: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]] = []
    winner_quote_id: Optional[str] = None
    total_savings_potential: float = 0.0
    analysis_complete: bool = False

class QuoteComparisonAgent:
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.comparison_agent = ConversableAgent(
            name="quote_comparison_expert",
            system_message="""You are an expert procurement analyst specializing in RFQ quote comparison and vendor evaluation.
            
            Your responsibilities:
            1. Analyze multiple vendor quotes for the same RFQ
            2. Perform comprehensive price, delivery, and compliance comparisons
            3. Identify cost-saving opportunities and risks
            4. Generate actionable recommendations for procurement decisions
            5. Create detailed comparison matrices and reports
            
            Always provide structured, data-driven analysis with clear justifications for recommendations.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        
        # In-memory storage for demo (replace with database later)
        self.quote_storage: Dict[str, List[VendorQuote]] = {}
        self.comparison_storage: Dict[str, QuoteComparison] = {}
        
    async def add_vendor_quote(self, rfq_id: str, quote_data: Dict[str, Any]) -> str:
        """Add a new vendor quote to the comparison system"""
        try:
            # Parse and validate quote data
            vendor_quote = self._parse_vendor_quote(rfq_id, quote_data)
            
            # Store quote
            if rfq_id not in self.quote_storage:
                self.quote_storage[rfq_id] = []
            
            self.quote_storage[rfq_id].append(vendor_quote)
            
            logger.info(f"Added quote {vendor_quote.quote_id} from {vendor_quote.vendor_name} for RFQ {rfq_id}")
            return vendor_quote.quote_id
            
        except Exception as e:
            logger.error(f"Failed to add vendor quote: {str(e)}")
            raise
    
    def _parse_vendor_quote(self, rfq_id: str, quote_data: Dict[str, Any]) -> VendorQuote:
        """Parse raw quote data into VendorQuote object"""
        line_items = []
        
        for item_data in quote_data.get('line_items', []):
            line_item = LineItemQuote(
                part_number=item_data.get('part_number', ''),
                description=item_data.get('description', ''),
                quantity=int(item_data.get('quantity', 0)),
                unit_price=float(item_data.get('unit_price', 0.0)),
                total_price=float(item_data.get('total_price', 0.0)),
                currency=item_data.get('currency', 'USD'),
                lead_time_days=item_data.get('lead_time_days'),
                compliance_score=float(item_data.get('compliance_score', 0.0)),
                notes=item_data.get('notes')
            )
            line_items.append(line_item)
        
        return VendorQuote(
            quote_id=str(uuid.uuid4()),
            vendor_name=quote_data.get('vendor_name', ''),
            vendor_email=quote_data.get('vendor_email', ''),
            rfq_id=rfq_id,
            submission_date=datetime.now(),
            line_items=line_items,
            total_quote_value=float(quote_data.get('total_quote_value', 0.0)),
            currency=quote_data.get('currency', 'USD'),
            delivery_terms=quote_data.get('delivery_terms'),
            payment_terms=quote_data.get('payment_terms'),
            validity_period_days=int(quote_data.get('validity_period_days', 30)),
            compliance_documents=quote_data.get('compliance_documents', []),
            vendor_rating=float(quote_data.get('vendor_rating', 0.0)),
            shipping_cost=float(quote_data.get('shipping_cost', 0.0)),
            tax_amount=float(quote_data.get('tax_amount', 0.0)),
            discount_percentage=float(quote_data.get('discount_percentage', 0.0))
        )
    
    async def compare_quotes(self, rfq_id: str, rfq_title: str = "") -> QuoteComparison:
        """Perform comprehensive quote comparison analysis"""
        
        if rfq_id not in self.quote_storage or len(self.quote_storage[rfq_id]) < 2:
            raise ValueError(f"Need at least 2 quotes for comparison. Found: {len(self.quote_storage.get(rfq_id, []))}")
        
        quotes = self.quote_storage[rfq_id]
        
        # Create comparison object
        comparison = QuoteComparison(
            rfq_id=rfq_id,
            rfq_title=rfq_title or f"RFQ-{rfq_id[:8]}",
            vendor_quotes=[asdict(quote) for quote in quotes]
        )
        
        # Generate comparison matrix
        comparison.comparison_matrix = await self._generate_comparison_matrix(quotes)
        
        # Get AI recommendations
        comparison.recommendations = await self._generate_ai_recommendations(quotes, comparison.comparison_matrix)
        
        # Determine winner and savings
        winner_analysis = self._determine_winner(quotes, comparison.comparison_matrix)
        comparison.winner_quote_id = winner_analysis['winner_quote_id']
        comparison.total_savings_potential = winner_analysis['savings_potential']
        
        comparison.analysis_complete = True
        
        # Store comparison
        self.comparison_storage[comparison.comparison_id] = comparison
        
        logger.info(f"Completed quote comparison for RFQ {rfq_id}. Winner: {winner_analysis['winner_vendor']}")
        
        return comparison
    
    async def _generate_comparison_matrix(self, quotes: List[VendorQuote]) -> Dict[str, Any]:
        """Generate detailed comparison matrix"""
        
        matrix = {
            'vendor_summary': [],
            'price_comparison': {},
            'delivery_comparison': {},
            'compliance_scores': {},
            'risk_assessment': {}
        }
        
        # Vendor summary
        for quote in quotes:
            matrix['vendor_summary'].append({
                'quote_id': quote.quote_id,
                'vendor_name': quote.vendor_name,
                'total_value': quote.total_quote_value,
                'currency': quote.currency,
                'line_items_count': len(quote.line_items),
                'vendor_rating': quote.vendor_rating,
                'submission_date': quote.submission_date.isoformat()
            })
        
        # Price comparison by line item
        if quotes:
            line_items = quotes[0].line_items
            for i, item in enumerate(line_items):
                part_number = item.part_number
                matrix['price_comparison'][part_number] = {}
                
                for quote in quotes:
                    if i < len(quote.line_items):
                        quote_item = quote.line_items[i]
                        matrix['price_comparison'][part_number][quote.vendor_name] = {
                            'unit_price': quote_item.unit_price,
                            'total_price': quote_item.total_price,
                            'currency': quote_item.currency,
                            'lead_time_days': quote_item.lead_time_days
                        }
        
        # Overall metrics
        total_values = [q.total_quote_value for q in quotes]
        matrix['price_comparison']['summary'] = {
            'lowest_total': min(total_values),
            'highest_total': max(total_values),
            'average_total': sum(total_values) / len(total_values),
            'price_range_percentage': ((max(total_values) - min(total_values)) / min(total_values)) * 100 if min(total_values) > 0 else 0
        }
        
        # Delivery comparison
        for quote in quotes:
            avg_lead_time = self._calculate_average_lead_time(quote.line_items)
            matrix['delivery_comparison'][quote.vendor_name] = {
                'average_lead_time_days': avg_lead_time,
                'delivery_terms': quote.delivery_terms,
                'shipping_cost': quote.shipping_cost
            }
        
        # Compliance scores
        for quote in quotes:
            avg_compliance = sum([item.compliance_score for item in quote.line_items]) / len(quote.line_items) if quote.line_items else 0
            matrix['compliance_scores'][quote.vendor_name] = {
                'average_compliance_score': avg_compliance,
                'documents_provided': len(quote.compliance_documents),
                'vendor_rating': quote.vendor_rating
            }
        
        return matrix
    
    def _calculate_average_lead_time(self, line_items: List[LineItemQuote]) -> Optional[float]:
        """Calculate average lead time for line items"""
        lead_times = [item.lead_time_days for item in line_items if item.lead_time_days is not None]
        return sum(lead_times) / len(lead_times) if lead_times else None
    
    async def _generate_ai_recommendations(self, quotes: List[VendorQuote], matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered recommendations"""
        
        analysis_prompt = f"""
        Analyze the following RFQ quote comparison data and provide procurement recommendations:

        QUOTES SUMMARY:
        {json.dumps([{
            'vendor': q.vendor_name,
            'total_value': q.total_quote_value,
            'currency': q.currency,
            'line_items': len(q.line_items),
            'vendor_rating': q.vendor_rating,
            'avg_lead_time': self._calculate_average_lead_time(q.line_items)
        } for q in quotes], indent=2)}

        COMPARISON MATRIX:
        {json.dumps(matrix, indent=2, default=str)}

        Please provide:
        1. Top recommendation with justification
        2. Risk assessment for each vendor
        3. Cost optimization opportunities
        4. Negotiation strategy suggestions
        5. Alternative recommendations if primary choice fails

        Return as JSON array with objects containing: type, priority, title, description, impact, action_required
        """
        
        try:
            response = self.comparison_agent.generate_reply([{
                "role": "user", 
                "content": analysis_prompt
            }])
            
            # Parse AI response
            recommendations = self._parse_ai_recommendations(response)
            return recommendations
            
        except Exception as e:
            logger.error(f"AI recommendation generation failed: {str(e)}")
            return self._generate_fallback_recommendations(quotes, matrix)
    
    def _parse_ai_recommendations(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured recommendations"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return [{"type": "analysis", "priority": "high", 
                        "title": "AI Analysis", "description": ai_response[:500],
                        "impact": "medium", "action_required": "review"}]
        except:
            return self._generate_basic_recommendations(ai_response)
    
    def _generate_fallback_recommendations(self, quotes: List[VendorQuote], matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic recommendations when AI fails"""
        recommendations = []
        
        # Price-based recommendation
        lowest_quote = min(quotes, key=lambda q: q.total_quote_value)
        highest_quote = max(quotes, key=lambda q: q.total_quote_value)
        
        savings = highest_quote.total_quote_value - lowest_quote.total_quote_value
        savings_percent = (savings / highest_quote.total_quote_value) * 100
        
        recommendations.append({
            "type": "cost_savings",
            "priority": "high",
            "title": f"Potential Savings: {savings_percent:.1f}%",
            "description": f"Selecting {lowest_quote.vendor_name} over {highest_quote.vendor_name} could save ${savings:,.2f}",
            "impact": "high" if savings_percent > 10 else "medium",
            "action_required": "negotiate_further" if savings_percent > 20 else "approve"
        })
        
        # Compliance recommendation
        best_compliance_quote = max(quotes, key=lambda q: sum(item.compliance_score for item in q.line_items))
        if best_compliance_quote != lowest_quote:
            recommendations.append({
                "type": "compliance_risk",
                "priority": "medium",
                "title": "Compliance vs Cost Trade-off",
                "description": f"{best_compliance_quote.vendor_name} has better compliance scores but higher cost",
                "impact": "medium",
                "action_required": "evaluate_risk"
            })
        
        return recommendations
    
    def _generate_basic_recommendations(self, ai_text: str) -> List[Dict[str, Any]]:
        """Generate basic recommendation from AI text"""
        return [{
            "type": "general",
            "priority": "medium",
            "title": "Analysis Summary",
            "description": ai_text[:300] + "..." if len(ai_text) > 300 else ai_text,
            "impact": "medium",
            "action_required": "review"
        }]
    
    def _determine_winner(self, quotes: List[VendorQuote], matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Determine winning quote using weighted scoring"""
        
        # Scoring weights (configurable)
        weights = {
            'price': 0.4,
            'delivery': 0.2,
            'compliance': 0.2,
            'vendor_rating': 0.2
        }
        
        scores = {}
        
        for quote in quotes:
            score = 0.0
            
            # Price score (inverse - lower price = higher score)
            total_values = [q.total_quote_value for q in quotes]
            min_price = min(total_values)
            max_price = max(total_values)
            if max_price > min_price:
                price_score = 1 - ((quote.total_quote_value - min_price) / (max_price - min_price))
            else:
                price_score = 1.0
            score += price_score * weights['price']
            
            # Delivery score
            avg_lead_time = self._calculate_average_lead_time(quote.line_items)
            if avg_lead_time is not None:
                all_lead_times = [self._calculate_average_lead_time(q.line_items) for q in quotes if self._calculate_average_lead_time(q.line_items) is not None]
                if all_lead_times:
                    min_lead = min(all_lead_times)
                    max_lead = max(all_lead_times)
                    if max_lead > min_lead:
                        delivery_score = 1 - ((avg_lead_time - min_lead) / (max_lead - min_lead))
                    else:
                        delivery_score = 1.0
                    score += delivery_score * weights['delivery']
            
            # Compliance score
            avg_compliance = sum([item.compliance_score for item in quote.line_items]) / len(quote.line_items) if quote.line_items else 0
            score += (avg_compliance / 100.0) * weights['compliance']  # Assuming compliance score is 0-100
            
            # Vendor rating score
            score += (quote.vendor_rating / 5.0) * weights['vendor_rating']  # Assuming rating is 0-5
            
            scores[quote.quote_id] = {
                'score': score,
                'vendor_name': quote.vendor_name,
                'total_value': quote.total_quote_value
            }
        
        # Find winner
        winner_quote_id = max(scores.keys(), key=lambda k: scores[k]['score'])
        winner_quote = next(q for q in quotes if q.quote_id == winner_quote_id)
        
        # Calculate savings potential
        all_values = [q.total_quote_value for q in quotes]
        max_value = max(all_values)
        savings_potential = max_value - winner_quote.total_quote_value
        
        return {
            'winner_quote_id': winner_quote_id,
            'winner_vendor': winner_quote.vendor_name,
            'winner_score': scores[winner_quote_id]['score'],
            'savings_potential': savings_potential,
            'all_scores': scores
        }
    
    def get_comparison_summary(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the comparison for reporting"""
        if comparison_id not in self.comparison_storage:
            return None
        
        comparison = self.comparison_storage[comparison_id]
        
        return {
            'comparison_id': comparison_id,
            'rfq_id': comparison.rfq_id,
            'rfq_title': comparison.rfq_title,
            'created_at': comparison.created_at.isoformat(),
            'vendor_count': len(comparison.vendor_quotes),
            'winner_quote_id': comparison.winner_quote_id,
            'total_savings_potential': comparison.total_savings_potential,
            'analysis_complete': comparison.analysis_complete,
            'recommendations_count': len(comparison.recommendations),
            'high_priority_recommendations': len([r for r in comparison.recommendations if r.get('priority') == 'high'])
        }
    
    def export_comparison_report(self, comparison_id: str, format: str = 'json') -> str:
        """Export comparison report in various formats"""
        if comparison_id not in self.comparison_storage:
            raise ValueError(f"Comparison {comparison_id} not found")
        
        comparison = self.comparison_storage[comparison_id]
        
        if format.lower() == 'json':
            return json.dumps(asdict(comparison), indent=2, default=str)
        elif format.lower() == 'csv':
            return self._export_csv_report(comparison)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv_report(self, comparison: QuoteComparison) -> str:
        """Export comparison as CSV format"""
        # Create summary data for CSV
        csv_data = []
        
        for vendor_data in comparison.vendor_quotes:
            csv_data.append({
                'Vendor': vendor_data['vendor_name'],
                'Total_Value': vendor_data['total_quote_value'],
                'Currency': vendor_data['currency'],
                'Line_Items': len(vendor_data['line_items']),
                'Vendor_Rating': vendor_data['vendor_rating'],
                'Status': vendor_data['status'],
                'Is_Winner': vendor_data['quote_id'] == comparison.winner_quote_id
            })
        
        # Convert to CSV string
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)

# Demo data generator for client presentations
class DemoDataGenerator:
    @staticmethod
    def generate_sample_rfq() -> Dict[str, Any]:
        """Generate sample RFQ for demo"""
        return {
            "rfq_id": "RFQ-2024-001",
            "title": "Industrial Sensors and Components",
            "client_name": "Manufacturing Corp",
            "line_items": [
                {
                    "part_number": "SENSOR-001",
                    "description": "Temperature Sensor -40°C to 150°C",
                    "quantity": 100,
                    "target_price": 45.00,
                    "currency": "USD"
                },
                {
                    "part_number": "VALVE-205",
                    "description": "Pneumatic Control Valve 1/4 inch",
                    "quantity": 50,
                    "target_price": 125.00,
                    "currency": "USD"
                }
            ]
        }
    
    @staticmethod
    def generate_sample_quotes(rfq_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sample vendor quotes for demo"""
        return [
            {
                "vendor_name": "TechSolutions Inc",
                "vendor_email": "quotes@techsolutions.com",
                "vendor_rating": 4.2,
                "total_quote_value": 16750.00,
                "currency": "USD",
                "delivery_terms": "FOB Destination",
                "payment_terms": "Net 30",
                "shipping_cost": 250.00,
                "line_items": [
                    {
                        "part_number": "SENSOR-001",
                        "description": "Temperature Sensor -40°C to 150°C",
                        "quantity": 100,
                        "unit_price": 42.50,
                        "total_price": 4250.00,
                        "currency": "USD",
                        "lead_time_days": 14,
                        "compliance_score": 95.0
                    },
                    {
                        "part_number": "VALVE-205", 
                        "description": "Pneumatic Control Valve 1/4 inch",
                        "quantity": 50,
                        "unit_price": 120.00,
                        "total_price": 6000.00,
                        "currency": "USD",
                        "lead_time_days": 21,
                        "compliance_score": 92.0
                    }
                ],
                "compliance_documents": ["ISO_9001", "Material_Cert", "Test_Report"]
            },
            {
                "vendor_name": "Global Components Ltd",
                "vendor_email": "sales@globalcomponents.com", 
                "vendor_rating": 3.8,
                "total_quote_value": 15900.00,
                "currency": "USD",
                "delivery_terms": "FOB Origin",
                "payment_terms": "Net 45",
                "shipping_cost": 400.00,
                "line_items": [
                    {
                        "part_number": "SENSOR-001",
                        "description": "Temperature Sensor -40°C to 150°C",
                        "quantity": 100,
                        "unit_price": 39.00,
                        "total_price": 3900.00,
                        "currency": "USD", 
                        "lead_time_days": 28,
                        "compliance_score": 88.0
                    },
                    {
                        "part_number": "VALVE-205",
                        "description": "Pneumatic Control Valve 1/4 inch", 
                        "quantity": 50,
                        "unit_price": 115.00,
                        "total_price": 5750.00,
                        "currency": "USD",
                        "lead_time_days": 35,
                        "compliance_score": 85.0
                    }
                ],
                "compliance_documents": ["ISO_9001", "Material_Cert"]
            },
            {
                "vendor_name": "Premium Industrial Supply",
                "vendor_email": "orders@premiumind.com",
                "vendor_rating": 4.7,
                "total_quote_value": 18200.00,
                "currency": "USD",
                "delivery_terms": "FOB Destination",
                "payment_terms": "Net 15",
                "shipping_cost": 0.00,
                "line_items": [
                    {
                        "part_number": "SENSOR-001",
                        "description": "Temperature Sensor -40°C to 150°C Premium Grade",
                        "quantity": 100,
                        "unit_price": 48.00,
                        "total_price": 4800.00,
                        "currency": "USD",
                        "lead_time_days": 7,
                        "compliance_score": 98.0
                    },
                    {
                        "part_number": "VALVE-205",
                        "description": "Pneumatic Control Valve 1/4 inch Premium",
                        "quantity": 50,
                        "unit_price": 135.00,
                        "total_price": 6750.00,
                        "currency": "USD",
                        "lead_time_days": 10,
                        "compliance_score": 97.0
                    }
                ],
                "compliance_documents": ["ISO_9001", "ISO_14001", "Material_Cert", "Test_Report", "Calibration_Cert"]
            }
        ]