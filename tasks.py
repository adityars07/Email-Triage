"""
tasks.py — Synthetic Email Dataset for the AI Email Triage Environment.

Contains 50 diverse emails with ground-truth labels for:
  - Classification (spam / ham)
  - Priority (low / medium / high / critical)
  - Reference reply (gold-standard response)
"""

import random
from typing import List, Dict, Optional


# ── Spam Emails (15) ─────────────────────────────────────────────────────────

SPAM_EMAILS = [
    {
        "id": "spam-001",
        "sender": "winner-notify@lotteryprizes.xyz",
        "subject": "🎉 You've Won $5,000,000!",
        "body": "Congratulations! You have been selected as the lucky winner of our international lottery. To claim your prize of $5,000,000 USD, please reply with your full name, address, and bank details. Act now — this offer expires in 24 hours!",
        "timestamp": "2025-03-15T08:23:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-002",
        "sender": "security-alert@paypai-verify.com",
        "subject": "Urgent: Your Account Has Been Compromised",
        "body": "Dear valued customer, we detected unusual activity on your PayPal account. Your account will be suspended unless you verify your identity within 12 hours. Click here to verify: http://paypai-verify.com/login. Regards, PayPal Security Team.",
        "timestamp": "2025-03-15T09:10:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-003",
        "sender": "dr.miracle@healthpills.net",
        "subject": "Lose 30 Pounds in 30 Days — Guaranteed!",
        "body": "Our revolutionary weight loss supplement has helped millions. No exercise needed! Order now and get 50% off. Limited time offer. Buy 2 get 1 free. Visit healthpills.net today. Unsubscribe link below.",
        "timestamp": "2025-03-15T10:45:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-004",
        "sender": "seo-guru@rankbooster.biz",
        "subject": "Get Your Website to #1 on Google — Today!",
        "body": "Hi, I noticed your website isn't ranking well. Our SEO experts can guarantee first-page Google results in just 7 days. We've helped 10,000+ businesses. Special discount for new clients — only $49/month. Reply YES to get started.",
        "timestamp": "2025-03-16T07:30:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-005",
        "sender": "prince.abubakar@diplomats.ng",
        "subject": "Confidential Business Proposal",
        "body": "Dear Friend, I am Prince Abubakar, son of the late General. I have $25 million trapped in a foreign account and need a trustworthy partner to help transfer it. You will receive 30% ($7.5M) for your assistance. Please respond with your full details urgently.",
        "timestamp": "2025-03-16T11:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-006",
        "sender": "deals@crypto-moonshot.io",
        "subject": "🚀 This Coin Will 1000x — Don't Miss Out!",
        "body": "Exclusive insider tip: MoonCoin ($MOON) is about to explode. Early investors are already seeing 500% returns. Buy now before it's too late. Join our VIP Telegram group for more signals. Not financial advice.",
        "timestamp": "2025-03-17T06:15:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-007",
        "sender": "noreply@cheap-watches-luxury.com",
        "subject": "Rolex, Omega, Cartier — 90% Off Today Only!",
        "body": "Premium luxury watches at unbeatable prices. Rolex Submariner — $199. Omega Seamaster — $149. Free shipping worldwide. Order now at cheap-watches-luxury.com. 100% authentic guaranteed.",
        "timestamp": "2025-03-17T14:20:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-008",
        "sender": "admin@microsoft-security-alert.xyz",
        "subject": "Your Microsoft 365 Subscription Has Expired",
        "body": "Your Microsoft 365 subscription expired on March 14, 2025. All your files on OneDrive will be deleted in 48 hours. Renew immediately at: http://microsoft-security-alert.xyz/renew. Microsoft Support Team.",
        "timestamp": "2025-03-18T09:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-009",
        "sender": "loans@quickcash-now.com",
        "subject": "Pre-Approved: $50,000 Personal Loan at 0% Interest",
        "body": "Congratulations! You've been pre-approved for a $50,000 personal loan with 0% APR for the first 12 months. No credit check required. Apply now — funds deposited within 24 hours. Call 1-800-FASHLOAN.",
        "timestamp": "2025-03-18T15:30:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-010",
        "sender": "pharmacy@discount-meds-online.ru",
        "subject": "Save 80% on Prescription Medications",
        "body": "Buy your medications online at a fraction of the cost. No prescription needed. Viagra, Cialis, Ambien — all available. Fast discreet shipping. Order at discount-meds-online.ru.",
        "timestamp": "2025-03-19T04:45:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-011",
        "sender": "gift-card@amazon-rewards.click",
        "subject": "Claim Your $500 Amazon Gift Card Now!",
        "body": "You have been chosen for a $500 Amazon Gift Card! Complete a short survey to claim your reward. Click here: http://amazon-rewards.click/survey. Offer valid for the next 3 hours only.",
        "timestamp": "2025-03-19T12:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-012",
        "sender": "careers@work-from-home-millions.com",
        "subject": "Earn $5,000/Week Working From Home!",
        "body": "Join thousands who are making $5,000+ per week from home! No experience required. Just 2 hours a day. We provide full training and support. Limited spots available — sign up at work-from-home-millions.com.",
        "timestamp": "2025-03-20T08:15:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-013",
        "sender": "support@apple-id-verify.net",
        "subject": "Action Required: Apple ID Locked",
        "body": "Your Apple ID has been locked due to suspicious activity. To unlock your account, verify your identity at http://apple-id-verify.net/unlock. If you do not verify within 24 hours, your account will be permanently disabled. Apple Support.",
        "timestamp": "2025-03-20T16:30:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-014",
        "sender": "marketing@bulk-email-service.biz",
        "subject": "Send 1 Million Emails for Just $99!",
        "body": "Reach millions of potential customers instantly. Our bulk email service delivers 1 million emails for only $99. High deliverability rate. No contracts. Get started today at bulk-email-service.biz.",
        "timestamp": "2025-03-21T10:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
    {
        "id": "spam-015",
        "sender": "dating@meet-singles-tonight.com",
        "subject": "Hot Singles in Your Area Want to Meet You!",
        "body": "Don't spend another night alone! Beautiful singles in your area are waiting to connect. Create your free profile now at meet-singles-tonight.com. 100% free — no credit card required.",
        "timestamp": "2025-03-21T22:45:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {"classification": "spam", "priority": "none", "reference_reply": ""},
    },
]

# ── Critical Priority Emails (5) ─────────────────────────────────────────────

CRITICAL_EMAILS = [
    {
        "id": "crit-001",
        "sender": "ops-alert@company.com",
        "subject": "🔴 CRITICAL: Production Database Server Down",
        "body": "ALERT: The primary production database (db-prod-01) is unreachable as of 03:15 UTC. All customer-facing services are returning 500 errors. Estimated impact: 50,000+ users affected. Failover did not trigger automatically. Immediate action required.",
        "timestamp": "2025-03-22T03:15:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "critical",
            "reference_reply": "Acknowledged. I'm investigating the database outage immediately. Will initiate manual failover to db-prod-02 and coordinate with the infrastructure team. I'll provide a status update within 15 minutes.",
        },
    },
    {
        "id": "crit-002",
        "sender": "ciso@company.com",
        "subject": "URGENT: Security Breach Detected — Customer Data Exposed",
        "body": "We've detected unauthorized access to our customer database. Preliminary analysis indicates that approximately 100,000 customer records may have been exposed, including names and email addresses. The attack vector appears to be an unpatched vulnerability in our API gateway. We need to assemble the incident response team immediately.",
        "timestamp": "2025-03-22T05:30:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "critical",
            "reference_reply": "This is extremely urgent. I'm joining the incident response immediately. I'll begin isolating the compromised API gateway and will coordinate with legal regarding breach notification requirements. Setting up a war room call in 10 minutes.",
        },
    },
    {
        "id": "crit-003",
        "sender": "ceo@company.com",
        "subject": "Board Meeting Moved to Today — Materials Needed ASAP",
        "body": "The board has called an emergency meeting for today at 2 PM regarding the acquisition. I need the updated financial projections, due diligence report, and competitive analysis within the next 2 hours. This is our highest priority right now. Cancel all other meetings.",
        "timestamp": "2025-03-22T08:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "critical",
            "reference_reply": "Understood. I'm prioritizing this immediately. I'll prepare the updated financial projections, compile the due diligence report, and finalize the competitive analysis. I'll have all materials ready and shared with you by 11:30 AM for your review before the board meeting.",
        },
    },
    {
        "id": "crit-004",
        "sender": "compliance@company.com",
        "subject": "REGULATORY DEADLINE: GDPR Audit Response Due Tomorrow",
        "body": "Reminder: Our response to the GDPR audit from the Data Protection Authority is due tomorrow at 5 PM CET. We are still missing the data processing inventory from three departments and the updated privacy impact assessments. Non-compliance could result in fines up to 4% of annual revenue. Please treat this as top priority.",
        "timestamp": "2025-03-22T09:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": True, "thread_length": 4},
        "ground_truth": {
            "classification": "ham",
            "priority": "critical",
            "reference_reply": "I understand the urgency. I'll personally follow up with the three remaining departments within the next hour to collect the data processing inventories. I'll also review and update the privacy impact assessments. I'll have everything consolidated and ready for your review by end of day today.",
        },
    },
    {
        "id": "crit-005",
        "sender": "devops@company.com",
        "subject": "CRITICAL: Payment Processing System Failure",
        "body": "Our payment processing pipeline has been failing for the last 45 minutes. No customer transactions are going through. The issue appears to be with our connection to the payment gateway. Revenue impact is approximately $15,000 per hour. Engineering team has been paged but we need management awareness and authorization for emergency vendor support escalation.",
        "timestamp": "2025-03-22T14:30:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "critical",
            "reference_reply": "Authorization granted for emergency vendor support escalation. Please contact the payment gateway's priority support line immediately. I'll notify finance and customer support teams about the outage. Keep me updated every 15 minutes until resolution.",
        },
    },
]

# ── High Priority Emails (10) ────────────────────────────────────────────────

HIGH_EMAILS = [
    {
        "id": "high-001",
        "sender": "vp.sales@company.com",
        "subject": "Client Escalation: Acme Corp Threatening to Cancel Contract",
        "body": "Acme Corp's VP of Operations called me directly today. They're very unhappy with the service outages over the past month and are considering canceling their $2M annual contract. They want a meeting with our executive team by Friday. We need to prepare a remediation plan and service credit proposal.",
        "timestamp": "2025-03-23T10:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "Thank you for flagging this. I'll prepare a remediation plan and service credit proposal by Thursday. Let's schedule an internal prep meeting for Wednesday afternoon to align on our approach before the Friday meeting with Acme Corp.",
        },
    },
    {
        "id": "high-002",
        "sender": "product.manager@company.com",
        "subject": "Feature Launch Deadline: Mobile App v3.0 — This Friday",
        "body": "Just a reminder that the v3.0 mobile app launch is scheduled for this Friday. QA has flagged 3 critical bugs that need to be fixed before release. The marketing campaign is already scheduled and ads go live on Saturday. We cannot delay this launch. Please prioritize the bug fixes.",
        "timestamp": "2025-03-23T11:30:00Z",
        "metadata": {"has_attachments": True, "is_reply": True, "thread_length": 6},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "I've reviewed the critical bugs from QA. I'll reassign two senior developers to focus exclusively on these fixes starting today. I'll provide a status update by end of day Wednesday. The Friday launch timeline should still be achievable.",
        },
    },
    {
        "id": "high-003",
        "sender": "hr.director@company.com",
        "subject": "Urgent: Key Employee Resignation — Engineering Lead",
        "body": "I wanted to inform you that Sarah Chen, our Engineering Lead, has submitted her resignation effective in two weeks. She's received an offer from a competitor. Given her critical role in the platform migration project, we should discuss retention options and succession planning immediately.",
        "timestamp": "2025-03-23T14:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "This is concerning given Sarah's role in the platform migration. Let's schedule a meeting today to discuss retention options — I'm available after 3 PM. In the meantime, please don't process the resignation yet. I'd like to have a conversation with Sarah first.",
        },
    },
    {
        "id": "high-004",
        "sender": "legal@company.com",
        "subject": "Contract Review Needed — $5M Partnership Deal",
        "body": "Attached is the final draft of the partnership agreement with TechGlobal Inc. The signing deadline is next Monday. There are a few clauses regarding intellectual property and liability that I'd like your input on before we proceed. Please review sections 4.2 and 7.1 specifically.",
        "timestamp": "2025-03-24T09:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "I'll review the contract today, focusing on sections 4.2 and 7.1 as you suggested. I'll send my comments and any proposed redlines by tomorrow morning so we have time for any negotiations before the Monday signing deadline.",
        },
    },
    {
        "id": "high-005",
        "sender": "finance@company.com",
        "subject": "Q1 Budget Overrun — Department Spending 20% Over Budget",
        "body": "Our Q1 analysis shows that the engineering department has exceeded its budget by 20%, primarily due to unplanned cloud infrastructure costs and contractor expenses. We need to present a corrective action plan to the CFO by next week. Can we meet to discuss cost reduction strategies?",
        "timestamp": "2025-03-24T11:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "Thank you for the heads up. I'll review the detailed spending breakdown and identify areas where we can reduce costs. Let's meet Thursday morning to discuss — I'll come prepared with a draft corrective action plan covering both infrastructure optimization and contractor consolidation.",
        },
    },
    {
        "id": "high-006",
        "sender": "customer.success@company.com",
        "subject": "Enterprise Client Onboarding Blocked — Integration Issues",
        "body": "GlobalBank's onboarding has been stalled for 5 days due to API integration failures. Their technical team reports that our SSO endpoints aren't responding correctly. This is a $800K ARR account and they have a hard go-live date of April 1st. We need engineering support urgently.",
        "timestamp": "2025-03-24T13:45:00Z",
        "metadata": {"has_attachments": False, "is_reply": True, "thread_length": 3},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "I'll assign a senior integration engineer to work directly with GlobalBank's team starting tomorrow. Let me also check our SSO endpoint logs to diagnose the issue beforehand. We should be able to unblock them within 48 hours to meet the April 1st deadline.",
        },
    },
    {
        "id": "high-007",
        "sender": "marketing.vp@company.com",
        "subject": "Press Release Approval Needed — Product Announcement Tomorrow",
        "body": "We have a major product announcement scheduled for tomorrow morning. The press release is attached and needs your final approval by 5 PM today. Several media outlets have been briefed under embargo. Any delays could damage our media relationships.",
        "timestamp": "2025-03-25T13:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "I'll review the press release right away and have my approval or feedback to you by 4 PM today, giving you time for any final adjustments. If I have concerns about specific claims, I'll call you directly.",
        },
    },
    {
        "id": "high-008",
        "sender": "infrastructure@company.com",
        "subject": "Data Center Migration — Downtime Window Confirmation Needed",
        "body": "We need your confirmation for the planned data center migration this weekend. The maintenance window is Saturday 11 PM to Sunday 6 AM EST. All non-critical services will be offline during this period. We need to send customer notifications by end of day tomorrow. Please confirm or suggest an alternative window.",
        "timestamp": "2025-03-25T15:30:00Z",
        "metadata": {"has_attachments": False, "is_reply": True, "thread_length": 5},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "The Saturday 11 PM to Sunday 6 AM EST window is confirmed. Please proceed with sending customer notifications. Make sure to include the support hotline number for any issues. I'll be available on-call during the migration window.",
        },
    },
    {
        "id": "high-009",
        "sender": "partner@techventures.com",
        "subject": "Investment Term Sheet — Response Needed by Friday",
        "body": "Following our productive discussions, we're pleased to present our investment term sheet for your Series B round. The proposed valuation is $150M with a $30M investment. Key terms are outlined in the attached document. We'd appreciate your response by this Friday to proceed to due diligence.",
        "timestamp": "2025-03-26T10:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "Thank you for the term sheet. We're excited about the potential partnership. I'll review the terms with our board and legal counsel and provide our response by Friday. If we have preliminary questions, we may reach out before then.",
        },
    },
    {
        "id": "high-010",
        "sender": "security@company.com",
        "subject": "Vulnerability Disclosure: Critical CVE in Auth Library",
        "body": "A critical vulnerability (CVE-2025-1234) has been disclosed in the authentication library we use (auth-lib v2.3.x). It allows remote code execution via crafted JWT tokens. A patch is available in v2.3.5. We need to update all services using this library within the next 48 hours.",
        "timestamp": "2025-03-26T16:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "high",
            "reference_reply": "Thank you for the alert. I'll initiate an audit of all services using auth-lib immediately. Let's plan an emergency patching sprint — I'll have the team update to v2.3.5 across all environments within 24 hours, starting with production-facing services.",
        },
    },
]

# ── Medium Priority Emails (10) ──────────────────────────────────────────────

MEDIUM_EMAILS = [
    {
        "id": "med-001",
        "sender": "pm.lead@company.com",
        "subject": "Weekly Sprint Review — Action Items from Today's Meeting",
        "body": "Hi team, here's a summary from today's sprint review. We completed 18 out of 22 story points. Carry-over items: user profile redesign, API rate limiter, and dashboard analytics. The next sprint planning is scheduled for Monday at 10 AM. Please update your JIRA tickets by Friday.",
        "timestamp": "2025-03-27T17:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Thanks for the summary. I'll update my JIRA tickets by Friday. For the carry-over items, I can take on the API rate limiter in the next sprint. See you at Monday's planning session.",
        },
    },
    {
        "id": "med-002",
        "sender": "training@company.com",
        "subject": "Upcoming Workshop: Cloud Architecture Best Practices — April 5",
        "body": "You're invited to a hands-on workshop on Cloud Architecture Best Practices on April 5th from 10 AM to 4 PM. Topics include microservices design, container orchestration, and cost optimization. Lunch will be provided. Please RSVP by March 30th.",
        "timestamp": "2025-03-27T09:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Thank you for the invitation. The workshop topics are very relevant to our current projects. I'd like to attend — please count me in. I'll add it to my calendar.",
        },
    },
    {
        "id": "med-003",
        "sender": "design.lead@company.com",
        "subject": "Design Review: New Dashboard Mockups Ready for Feedback",
        "body": "Hi, the design team has completed the new dashboard mockups based on the user research findings. I've attached the Figma link and a PDF export. We'd love your feedback by next Wednesday so we can finalize the designs and hand them off to engineering.",
        "timestamp": "2025-03-28T11:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Thanks for sharing the mockups. I'll review the designs and provide detailed feedback by Wednesday. At first glance, they look great. I may have some suggestions about the data visualization components.",
        },
    },
    {
        "id": "med-004",
        "sender": "vendor@cloudprovider.com",
        "subject": "Your Monthly Cloud Usage Report — March 2025",
        "body": "Your March 2025 cloud usage report is ready. Total spend: $24,350 (up 8% from February). Top services: Compute ($12,100), Storage ($5,200), Database ($4,800). We recommend reviewing your reserved instance coverage — you could save up to $3,000/month. See the attached detailed report.",
        "timestamp": "2025-03-31T08:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Thank you for the report. The 8% increase is noted — I'll review the detailed breakdown and evaluate the reserved instance recommendations. Could you schedule a call next week to discuss optimization strategies?",
        },
    },
    {
        "id": "med-005",
        "sender": "team.lead@company.com",
        "subject": "Request: Quarterly Performance Reviews Due April 15",
        "body": "Reminder that quarterly performance reviews for your direct reports are due by April 15th. Please complete the review forms in our HR portal. If you need the self-assessment forms from your team members, please remind them — the deadline for self-assessments is April 10th.",
        "timestamp": "2025-03-31T10:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Thanks for the reminder. I'll send out self-assessment reminders to my team today with the April 10th deadline. I'll have all performance reviews completed and submitted in the HR portal by April 15th.",
        },
    },
    {
        "id": "med-006",
        "sender": "colleague@company.com",
        "subject": "Can You Review My Pull Request? — Auth Module Refactor",
        "body": "Hey, I've submitted a PR for the auth module refactor (#1247). It's about 400 lines of changes — mostly restructuring the middleware chain and adding refresh token rotation. Would you be able to review it sometime this week? No rush, but I'd like to merge before next sprint.",
        "timestamp": "2025-04-01T14:30:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Sure, I'll take a look at PR #1247. I should be able to review it by Thursday. The auth module refactor sounds important — I'll pay special attention to the refresh token rotation logic.",
        },
    },
    {
        "id": "med-007",
        "sender": "facilities@company.com",
        "subject": "Office Renovation: 3rd Floor Kitchen Closed April 7-11",
        "body": "Please be advised that the 3rd floor kitchen will be closed from April 7-11 for renovation. During this time, please use the 2nd floor kitchen or the cafeteria on the ground floor. The renovated kitchen will feature new appliances and an expanded seating area. We apologize for the inconvenience.",
        "timestamp": "2025-04-01T09:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Thanks for the heads up. I'll let my team know about the kitchen closure and the alternatives. Looking forward to the renovated space!",
        },
    },
    {
        "id": "med-008",
        "sender": "data.analyst@company.com",
        "subject": "Updated KPI Dashboard — Q1 Metrics Ready",
        "body": "I've updated the KPI dashboard with Q1 2025 metrics. Key highlights: MAU up 15%, customer churn down to 2.3%, NPS increased to 72. There are a few data quality issues with the revenue attribution model that I'm still working on. Dashboard link is in the shared workspace.",
        "timestamp": "2025-04-02T10:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": True, "thread_length": 2},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "Great work on the Q1 dashboard. The metrics look encouraging, especially the churn improvement. Let me know when the revenue attribution issues are resolved — I'd like to include those numbers in the board update.",
        },
    },
    {
        "id": "med-009",
        "sender": "intern.coordinator@company.com",
        "subject": "Summer Internship Program — Mentor Volunteers Needed",
        "body": "Our summer internship program starts June 1st and we're looking for mentor volunteers. Each mentor would be paired with one intern for 12 weeks. Time commitment is approximately 2-3 hours per week for mentoring sessions and code reviews. Please reply if you're interested.",
        "timestamp": "2025-04-02T11:30:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "I'd be happy to volunteer as a mentor for the summer internship program. The 2-3 hour weekly commitment works for my schedule. Please sign me up and let me know the next steps.",
        },
    },
    {
        "id": "med-010",
        "sender": "tech.writer@company.com",
        "subject": "API Documentation Update — Review Requested",
        "body": "I've completed the updated API documentation for the v3 endpoints. Changes include new authentication flows, webhook configurations, and rate limit details. Could you review the technical accuracy? The draft is in Confluence. Deadline for final review is April 10th.",
        "timestamp": "2025-04-03T09:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "medium",
            "reference_reply": "I'll review the API documentation for technical accuracy. I'll focus on the authentication flows and rate limit details since those are areas I'm most familiar with. Expect my feedback by April 8th.",
        },
    },
]

# ── Low Priority Emails (10) ─────────────────────────────────────────────────

LOW_EMAILS = [
    {
        "id": "low-001",
        "sender": "newsletter@techweekly.com",
        "subject": "Tech Weekly: AI Trends, Cloud Updates & More",
        "body": "This week in tech: OpenAI launches new reasoning model, AWS announces price cuts for S3, Google releases Flutter 5.0, and Microsoft expands GitHub Copilot features. Read the full articles on our website. Unsubscribe from this newsletter.",
        "timestamp": "2025-04-03T07:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Thanks for the newsletter. The AWS S3 price cuts are interesting — I'll look into how that affects our infrastructure costs.",
        },
    },
    {
        "id": "low-002",
        "sender": "social.committee@company.com",
        "subject": "Friday Happy Hour — Rooftop Bar, 5 PM",
        "body": "Join us this Friday at 5 PM for happy hour at the rooftop bar! First round of drinks is on the company. It's a great chance to catch up with colleagues from other teams. RSVP is optional but appreciated for headcount. See you there!",
        "timestamp": "2025-04-03T12:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Sounds fun! I'll try to make it. Count me in for the headcount.",
        },
    },
    {
        "id": "low-003",
        "sender": "colleague.jane@company.com",
        "subject": "FYI: Interesting Article on Microservices Patterns",
        "body": "Hey, I came across this great article on microservices communication patterns. Thought you might find it interesting given our current architecture discussions. No action needed — just sharing for your reading list. Link: https://techblog.example.com/microservices-patterns",
        "timestamp": "2025-04-04T15:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Thanks for sharing, Jane! I'll add it to my reading list. Always good to see different perspectives on microservices patterns.",
        },
    },
    {
        "id": "low-004",
        "sender": "it-support@company.com",
        "subject": "Scheduled Maintenance: Email Server — Sunday 2-4 AM",
        "body": "Routine maintenance will be performed on the email server this Sunday from 2 AM to 4 AM EST. Email service may be briefly interrupted during this window. No action required on your part. Thank you for your patience.",
        "timestamp": "2025-04-04T10:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Noted, thanks for the heads up on the maintenance window.",
        },
    },
    {
        "id": "low-005",
        "sender": "wellness@company.com",
        "subject": "April Wellness Challenge: 10,000 Steps a Day",
        "body": "Join our April Wellness Challenge! Track your daily steps and aim for 10,000 steps per day. Participants who complete the challenge will receive a company wellness kit. Sign up through the wellness portal by April 5th. Stay healthy!",
        "timestamp": "2025-04-01T08:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "This sounds like a great initiative! I'll sign up through the wellness portal. Thanks for organizing this.",
        },
    },
    {
        "id": "low-006",
        "sender": "parking@company.com",
        "subject": "Parking Lot Resurfacing — Use East Lot April 14-16",
        "body": "The main parking lot will be resurfaced from April 14-16. During this time, please use the East parking lot. Shuttle service will be available from the East lot to the main entrance every 10 minutes from 7 AM to 7 PM. Thank you for your cooperation.",
        "timestamp": "2025-04-05T09:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Thanks for the notice. I'll plan to use the East lot those days.",
        },
    },
    {
        "id": "low-007",
        "sender": "learning@company.com",
        "subject": "New Courses Available on Learning Platform",
        "body": "We've added 15 new courses to our learning platform including: Advanced Python, System Design Interview Prep, Leadership Fundamentals, and Data Engineering with Spark. Browse the full catalog at learning.company.com. Remember, you have a $1,500 annual learning budget.",
        "timestamp": "2025-04-05T11:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Thanks for sharing the new courses. I'm interested in the System Design and Data Engineering courses. I'll check them out on the platform.",
        },
    },
    {
        "id": "low-008",
        "sender": "green.team@company.com",
        "subject": "Earth Day Initiative: Office Plant Adoption Program",
        "body": "In celebration of Earth Day, we're launching an office plant adoption program! Each team can adopt a plant for their workspace. Plants improve air quality and boost productivity. Sign up by April 18th. Succulents, pothos, and snake plants are available.",
        "timestamp": "2025-04-06T10:00:00Z",
        "metadata": {"has_attachments": True, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Love this idea! Our team would like to adopt a snake plant for our workspace. I'll coordinate with the team and sign up by the deadline.",
        },
    },
    {
        "id": "low-009",
        "sender": "alumni@university.edu",
        "subject": "Alumni Newsletter — Spring 2025 Edition",
        "body": "Greetings from your alma mater! In this edition: Campus expansion updates, notable alumni achievements, upcoming reunion events, and scholarship fund progress. We've also launched a new mentorship program connecting current students with alumni. Read more on our website.",
        "timestamp": "2025-04-06T14:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Thanks for the update. The mentorship program sounds interesting — I'd be happy to participate. Please send me more details about how to sign up as a mentor.",
        },
    },
    {
        "id": "low-010",
        "sender": "book.club@company.com",
        "subject": "April Book Club Pick: 'Designing Data-Intensive Applications'",
        "body": "Our April book club pick is 'Designing Data-Intensive Applications' by Martin Kleppmann. Discussion meeting is April 28th at noon in the common area. The company will reimburse the book purchase. Reading is optional — all are welcome to join the discussion!",
        "timestamp": "2025-04-07T09:00:00Z",
        "metadata": {"has_attachments": False, "is_reply": False, "thread_length": 1},
        "ground_truth": {
            "classification": "ham",
            "priority": "low",
            "reference_reply": "Great pick! I've been wanting to read that book. I'll get a copy and join the discussion on April 28th.",
        },
    },
]


# ── Task Pool ─────────────────────────────────────────────────────────────────

ALL_EMAILS = SPAM_EMAILS + CRITICAL_EMAILS + HIGH_EMAILS + MEDIUM_EMAILS + LOW_EMAILS


class TaskPool:
    """Manages the pool of email tasks for the environment."""

    def __init__(self, emails: Optional[List[Dict]] = None, shuffle: bool = True):
        self.emails = list(emails or ALL_EMAILS)
        self._original = list(self.emails)
        self._index = 0
        if shuffle:
            random.shuffle(self.emails)

    def sample(self) -> Dict:
        """Return the next email in the pool (cycles through all)."""
        if self._index >= len(self.emails):
            self._index = 0
            random.shuffle(self.emails)
        email = self.emails[self._index]
        self._index += 1
        return email

    def get_by_id(self, email_id: str) -> Optional[Dict]:
        """Retrieve a specific email by its ID."""
        for email in self._original:
            if email["id"] == email_id:
                return email
        return None

    def get_all(self) -> List[Dict]:
        """Return all emails in original order."""
        return list(self._original)

    def filter_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Filter emails by difficulty level.

        - easy   → spam emails
        - medium → low + medium priority
        - hard   → high + critical priority
        """
        mapping = {
            "easy": SPAM_EMAILS,
            "medium": MEDIUM_EMAILS + LOW_EMAILS,
            "hard": HIGH_EMAILS + CRITICAL_EMAILS,
        }
        return list(mapping.get(difficulty, self._original))

    def __len__(self) -> int:
        return len(self.emails)

    def reset(self, shuffle: bool = True):
        """Reset the pool index and optionally re-shuffle."""
        self._index = 0
        self.emails = list(self._original)
        if shuffle:
            random.shuffle(self.emails)
