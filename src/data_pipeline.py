import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import re

class BiotechTrialPipeline:
    """
    Fetches clinical trial data from ClinicalTrials.gov API and maps to stock tickers.
    
    CRITICAL: The hardest part isn't the API - it's mapping sponsor names to tickers.
    ClinicalTrials.gov uses company legal names ('Intra-Cellular Therapies, Inc.')
    while stock tickers are symbols ('ITCI'). You need a mapping file.
    """
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.trials_data = []
        
    def fetch_trials_by_condition(self, condition="cancer", phase="PHASE3", max_results=100):
        """
        Pulls trials from ClinicalTrials.gov API.
        
        :param condition: Disease area (e.g., 'cancer', 'alzheimer', 'diabetes')
        :param phase: Trial phase filter ('PHASE1', 'PHASE2', 'PHASE3', 'PHASE4')
        :param max_results: Number of trials to fetch (API limit: 1000 per request)
        """
        print(f"Fetching {phase} {condition} trials from ClinicalTrials.gov...")
        
        params = {
            "query.cond": condition,
            "query.phase": phase,
            "filter.overallStatus": "COMPLETED,ACTIVE_NOT_RECRUITING,RECRUITING",
            "pageSize": min(max_results, 1000),
            "format": "json"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            studies = data.get('studies', [])
            print(f"Retrieved {len(studies)} trials.")
            
            # Extract relevant fields
            for study in studies:
                protocol = study.get('protocolSection', {})
                
                # Identification
                nct_id = protocol.get('identificationModule', {}).get('nctId', 'N/A')
                title = protocol.get('identificationModule', {}).get('officialTitle', 'N/A')
                
                # Sponsor (this is what we'll map to ticker)
                sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
                lead_sponsor = sponsor_module.get('leadSponsor', {}).get('name', 'N/A')
                
                # Dates - THIS IS YOUR EVENT DATE
                status_module = protocol.get('statusModule', {})
                completion_date = status_module.get('primaryCompletionDateStruct', {}).get('date', None)
                
                # If no primary completion, try study completion
                if not completion_date:
                    completion_date = status_module.get('completionDateStruct', {}).get('date', None)
                
                # Phase verification
                design_module = protocol.get('designModule', {})
                phases = design_module.get('phases', [])
                
                # Store
                self.trials_data.append({
                    'nct_id': nct_id,
                    'title': title,
                    'sponsor': lead_sponsor,
                    'completion_date': completion_date,
                    'phase': ', '.join(phases) if phases else 'N/A',
                    'status': status_module.get('overallStatus', 'N/A')
                })
                
            return pd.DataFrame(self.trials_data)
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return pd.DataFrame()
    
    def map_sponsors_to_tickers(self, trials_df, mapping_file='sponsor_ticker_map.csv'):
        """
        CRITICAL FUNCTION: Maps sponsor company names to stock tickers.
        
        The reality: This is the hardest part. ClinicalTrials.gov has sponsor names like:
        - 'Eli Lilly and Company'
        - 'Pfizer'
        - 'Intra-Cellular Therapies, Inc.'
        
        You need a manual mapping file. Here's why automated mapping fails:
        - yfinance ticker search is unreliable for exact matching
        - Many biotech companies have subsidiaries or parent companies
        - Private companies have trials but no ticker
        
        RECOMMENDATION: Create 'sponsor_ticker_map.csv' with columns: [sponsor, ticker]
        Example:
        sponsor,ticker
        Eli Lilly and Company,LLY
        Pfizer,PFE
        Intra-Cellular Therapies Inc,ITCI
        """
        try:
            # Load your manual mapping file
            mapping_df = pd.read_csv(mapping_file)
            print(f"Loaded {len(mapping_df)} sponsor-to-ticker mappings.")
            
            # Merge with trials data
            merged = trials_df.merge(
                mapping_df, 
                left_on='sponsor', 
                right_on='sponsor', 
                how='left'
            )
            
            # Filter out trials without ticker matches (private companies, non-US, etc.)
            matched = merged[merged['ticker'].notna()].copy()
            
            print(f"Matched {len(matched)} out of {len(trials_df)} trials to tickers.")
            print(f"Unmatched sponsors: {merged[merged['ticker'].isna()]['sponsor'].unique()[:10]}")
            
            return matched
            
        except FileNotFoundError:
            print(f"ERROR: {mapping_file} not found.")
            print("You must create a CSV file mapping sponsor names to tickers.")
            print("See function docstring for format.")
            return pd.DataFrame()
    
    def filter_upcoming_events(self, trials_df, days_ahead=180):
        """
        Filters for trials completing in the next X days.
        These are your 'tradeable catalysts'.
        """
        trials_df['completion_date'] = pd.to_datetime(trials_df['completion_date'], errors='coerce')
        
        today = pd.Timestamp.now()
        future_cutoff = today + pd.Timedelta(days=days_ahead)
        
        upcoming = trials_df[
            (trials_df['completion_date'] >= today) & 
            (trials_df['completion_date'] <= future_cutoff)
        ].copy()
        
        print(f"Found {len(upcoming)} trials completing in next {days_ahead} days.")
        return upcoming.sort_values('completion_date')
    
    def export_for_backtester(self, trials_df, output_file='trial_events.csv'):
        """
        Formats data for your BiotechEventBacktester class.
        Outputs: [ticker, event_date, quality_score, catalyst_type]
        
        NOTE: 'quality_score' comes from YOUR BioTrial Quality Analyzer.
        This function creates a placeholder. You'll replace this with actual scores.
        """
        # Format for backtester
        formatted = trials_df[['ticker', 'completion_date', 'nct_id', 'phase']].copy()
        formatted.columns = ['ticker', 'event_date', 'trial_id', 'catalyst_type']
        
        # PLACEHOLDER: You will replace this with your BioTrial Analyzer scores
        formatted['quality_score'] = 'NEEDS_ANALYSIS'  # <-- THIS IS WHERE YOUR TOOL GOES
        
        formatted.to_csv(output_file, index=False)
        print(f"Exported {len(formatted)} events to {output_file}")
        print("\n⚠️  NEXT STEP: Run your BioTrial Quality Analyzer on these NCT IDs")
        print("to populate the 'quality_score' column with 'High' or 'Low' ratings.")
        
        return formatted

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = BiotechTrialPipeline()
    
    # 1. Fetch Phase 3 oncology trials
    trials = pipeline.fetch_trials_by_condition(
        condition="oncology", 
        phase="PHASE3", 
        max_results=50
    )
    
    # 2. Map to tickers (requires manual mapping file)
    # You need to create 'sponsor_ticker_map.csv' first!
    matched_trials = pipeline.map_sponsors_to_tickers(trials)
    
    # 3. Filter for upcoming catalysts (next 6 months)
    upcoming = pipeline.filter_upcoming_events(matched_trials, days_ahead=180)
    
    # 4. Export for backtesting
    if not upcoming.empty:
        pipeline.export_for_backtester(upcoming, output_file='biotech_catalysts_2026.csv')
