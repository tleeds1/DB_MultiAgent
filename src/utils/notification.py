"""
Notification management system
"""

import os
import requests
import re
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NotificationManager:
    """Manage notifications across different channels"""
    
    def __init__(self):
        self.channels = {
            'telegram': {
                'token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID')
            },
            'slack': {
                'webhook': os.getenv('SLACK_WEBHOOK')
            },
            'whatsapp': {
                'account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
                'auth_token': os.getenv('TWILIO_AUTH_TOKEN'),
                'from_number': os.getenv('TWILIO_WHATSAPP_FROM'),
                'to_number': os.getenv('TWILIO_WHATSAPP_TO')
            }
        }
    
    def send_notification(self, message: str, channel: str = 'telegram', 
                         notification_type: str = "INFO", **kwargs) -> bool:
        """Send notification to specified channel"""
        try:
            if channel == 'telegram':
                return self._send_telegram(message, notification_type)
            elif channel == 'slack':
                return self._send_slack(message, notification_type)
            elif channel == 'whatsapp':
                return self._send_whatsapp(message)
            elif channel == 'all':
                success = True
                for ch in ['telegram', 'slack', 'whatsapp']:
                    if not self._send_to_channel(message, ch, notification_type):
                        success = False
                return success
            else:
                print(f"Unknown notification channel: {channel}")
                return False
        except Exception as e:
            print(f"Notification error: {e}")
            return False
    
    def _send_to_channel(self, message: str, channel: str, notification_type: str) -> bool:
        """Send to specific channel"""
        if channel == 'telegram':
            return self._send_telegram(message, notification_type)
        elif channel == 'slack':
            return self._send_slack(message, notification_type)
        elif channel == 'whatsapp':
            return self._send_whatsapp(message)
        return False
    
    def _send_telegram(self, message: str, notification_type: str = "INFO") -> bool:
        """Send Telegram notification with proper formatting"""
        if not self.channels['telegram']['token'] or not self.channels['telegram']['chat_id']:
            print("Telegram configuration missing")
            return False
            
        token = self.channels['telegram']['token']
        chat_id = self.channels['telegram']['chat_id']
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        # Add notification type emoji
        type_emoji = {
            "SUCCESS": "✅",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "INFO": "ℹ️"
        }
        
        final_message = f"{type_emoji.get(notification_type, 'ℹ️')} {message}"
        
        # Truncate long messages
        if len(final_message) > 4000:
            final_message = final_message[:4000] + "\n\n... (truncated)"
        
        # Escape markdown for Telegram
        clean_message = self._escape_telegram_markdown(final_message)
        
        payload = {
            "chat_id": chat_id,
            "text": clean_message,
            "parse_mode": "MarkdownV2"
        }
        
        try:
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code != 200:
                print(f"Telegram notification failed: {response.text}")
                # Fallback without markdown
                payload["parse_mode"] = None
                payload["text"] = final_message
                response = requests.post(url, data=payload, timeout=10)
                return response.status_code == 200
            return True
        except Exception as e:
            print(f"Telegram notification error: {e}")
            return False
    
    def _send_slack(self, message: str, notification_type: str = "INFO") -> bool:
        """Send Slack notification"""
        if not self.channels['slack']['webhook']:
            print("Slack configuration missing")
            return False
            
        # Add notification type emoji
        type_emoji = {
            "SUCCESS": "✅",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "INFO": "ℹ️"
        }
        
        final_message = f"{type_emoji.get(notification_type, 'ℹ️')} {message}"
        
        payload = {
            "text": final_message,
            "username": "Database Agent",
            "icon_emoji": ":robot_face:",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": final_message
                    }
                }
            ]
        }
        
        try:
            response = requests.post(
                self.channels['slack']['webhook'], 
                json=payload, 
                timeout=10
            )
            if response.status_code != 200:
                print(f"Slack notification failed: {response.text}")
                return False
            return True
        except Exception as e:
            print(f"Slack notification error: {e}")
            return False

    def _send_whatsapp(self, message: str) -> bool:
        """Send WhatsApp message via Twilio API"""
        cfg = self.channels['whatsapp']
        if not all([cfg.get('account_sid'), cfg.get('auth_token'), cfg.get('from_number'), cfg.get('to_number')]):
            # Config missing; silently skip
            return False
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{cfg['account_sid']}/Messages.json"
            data = {
                'From': f"whatsapp:{cfg['from_number']}",
                'To': f"whatsapp:{cfg['to_number']}",
                'Body': message[:1590]  # WhatsApp body limit safety
            }
            resp = requests.post(url, data=data, auth=(cfg['account_sid'], cfg['auth_token']), timeout=10)
            return resp.status_code in (200, 201)
        except Exception as e:
            print(f"WhatsApp notification error: {e}")
            return False
    
    def _escape_telegram_markdown(self, text: str) -> str:
        """Escape markdown for Telegram"""
        escape_chars = r'_*\[\]()~`>#+-=|{}.!'
        
        def escape_markdown_v2_except_codeblocks(text):
            parts = re.split(r'(```[\s\S]*?```)', text)
            for i, part in enumerate(parts):
                if not part.startswith('```'):
                    parts[i] = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', part)
            return ''.join(parts)
        
        return escape_markdown_v2_except_codeblocks(text)
    
    def create_structured_notification(self, question: str, sql: str, results: Dict[str, Any], 
                                     execution_time: float, context: Dict[str, Any], 
                                     status: str = "SUCCESS") -> str:
        """Create structured notification in requested format"""
        
        # Determine if database was changed
        database_changed = "No"
        changes_description = ""
        
        # Check if this was a write operation based on available fields
        is_write_operation = (
            results and (
                results.get('is_write_operation', False) or
                results.get('affected_rows', 0) > 0 or
                'affected_rows' in results
            )
        )
        
        if is_write_operation:
            database_changed = "Yes"
            operation_type = results.get('operation_type', 'UNKNOWN')
            affected_rows = results.get('affected_rows', 0)
            
            if operation_type == 'INSERT':
                changes_description = f"Inserted {affected_rows} new records"
            elif operation_type == 'UPDATE':
                changes_description = f"Updated {affected_rows} existing records"
            elif operation_type == 'DELETE':
                changes_description = f"Deleted {affected_rows} records"
            elif operation_type in ['CREATE', 'ALTER', 'DROP']:
                changes_description = f"Schema modification: {operation_type} operation"
            else:
                changes_description = f"Modified {affected_rows} rows"
        
        # Create short analysis
        analysis = self._generate_short_analysis(results, context, question)
        
        # Format the notification with results
        notification = f"""
🤖 **Database Agent Report**

**Question:** {question[:150]}{'...' if len(question) > 150 else ''}

**SQL Query:**
```sql
{sql}
```

**Result:** {self._format_result_summary(results)}

**Changes made to database:** {database_changed}
{f"({changes_description})" if changes_description else ""}

**Analysis:** {analysis}

---
⚡ Execution: {execution_time:.2f}s | 🔄 Status: {status}
"""
        
        # Add actual query results if available
        if results and 'rows' in results and results['rows']:
            notification += f"\n**Query Results:**\n"
            notification += self._format_results_for_notification(results)
        
        return notification
    
    def _format_result_summary(self, results: Dict[str, Any]) -> str:
        """Format result summary for notification"""
        if not results:
            return "No results"
        
        if 'rows' in results:
            row_count = results.get('row_count', 0)
            columns = len(results.get('columns', []))
            return f"Found {row_count} records with {columns} columns"
        elif 'affected_rows' in results:
            affected = results.get('affected_rows', 0)
            operation = results.get('operation_type', 'MODIFY')
            return f"{operation} completed - {affected} rows affected"
        else:
            return "Operation completed successfully"
    
    def _generate_short_analysis(self, results: Dict[str, Any], context: Dict[str, Any], question: str) -> str:
        """Generate concise analysis for notification"""
        
        if not results:
            return "Query execution failed - check logs for details"
        
        analysis_points = []
        
        # Performance analysis
        exec_time = results.get('execution_time', 0)
        if exec_time > 2.0:
            analysis_points.append("⚠️ Long execution time detected")
        elif exec_time < 0.1:
            analysis_points.append("⚡ Fast query execution")
        
        # Data analysis
        if 'row_count' in results:
            row_count = results['row_count']
            if row_count == 0:
                analysis_points.append("❗ No matching records found")
            elif row_count > 1000:
                analysis_points.append("📊 Large dataset returned")
            else:
                analysis_points.append(f"✅ Retrieved {row_count} records")
        
        # Operation analysis
        if results.get('is_write_operation'):
            affected = results.get('affected_rows', 0)
            if affected > 0:
                analysis_points.append("✅ Database successfully modified")
            else:
                analysis_points.append("⚠️ Write operation but no rows affected")
        
        # Context-based analysis
        if context.get('complexity') == 'complex':
            analysis_points.append("🔧 Complex query executed successfully")
        
        return " | ".join(analysis_points) if analysis_points else "Standard query execution completed"
    
    def _format_results_for_notification(self, results: Dict[str, Any]) -> str:
        """Format query results for notification display"""
        if not results or 'rows' not in results or not results['rows']:
            return "No results to display"
        
        formatted_results = []
        columns = results.get('columns', [])
        rows = results.get('rows', [])
        row_count = results.get('row_count', 0)
        
        # Limit results for notification (avoid very long messages)
        max_rows = min(5, len(rows))  # Show max 5 rows in notification
        
        if columns:
            # Create header
            header = " | ".join(str(col) for col in columns)
            formatted_results.append(header)
            formatted_results.append("-" * len(header))
        
        # Add rows
        for i, row in enumerate(rows[:max_rows]):
            formatted_row = " | ".join(str(val) for val in row)
            formatted_results.append(formatted_row)
        
        # Add summary if there are more rows
        if row_count > max_rows:
            formatted_results.append(f"... and {row_count - max_rows} more rows")
        
        return "\n".join(formatted_results)
