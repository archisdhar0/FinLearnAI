-- ============================================================================
-- Seed Leaderboard Data for FinLearn AI
-- Run this in Supabase SQL Editor to populate sample users and scores
-- ============================================================================

-- First, insert sample user profiles
INSERT INTO user_profiles (user_id, display_name, created_at, updated_at)
VALUES
  ('sample-user-001', 'InvestorPro', NOW() - INTERVAL '60 days', NOW()),
  ('sample-user-002', 'WealthBuilder', NOW() - INTERVAL '50 days', NOW()),
  ('sample-user-003', 'StockSavvy', NOW() - INTERVAL '55 days', NOW()),
  ('sample-user-004', 'MarketMaster', NOW() - INTERVAL '40 days', NOW()),
  ('sample-user-005', 'DividendKing', NOW() - INTERVAL '45 days', NOW()),
  ('sample-user-006', 'IndexFundFan', NOW() - INTERVAL '35 days', NOW()),
  ('sample-user-007', 'CompoundKing', NOW() - INTERVAL '30 days', NOW()),
  ('sample-user-008', 'RetireEarly', NOW() - INTERVAL '25 days', NOW()),
  ('sample-user-009', 'BudgetBoss', NOW() - INTERVAL '20 days', NOW()),
  ('sample-user-010', 'NewInvestor', NOW() - INTERVAL '15 days', NOW()),
  ('sample-user-011', 'SavingsGuru', NOW() - INTERVAL '10 days', NOW()),
  ('sample-user-012', 'FinanceFresh', NOW() - INTERVAL '5 days', NOW())
ON CONFLICT (user_id) DO UPDATE SET display_name = EXCLUDED.display_name;

-- Insert lesson quiz scores for sample users
-- User 1: InvestorPro - High performer
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-001', 'foundation', 'what-is-investing', 95, 95, NOW() - INTERVAL '58 days'),
  ('sample-user-001', 'foundation', 'stocks-bonds-funds', 92, 92, NOW() - INTERVAL '56 days'),
  ('sample-user-001', 'foundation', 'how-markets-work', 98, 98, NOW() - INTERVAL '54 days'),
  ('sample-user-001', 'foundation', 'time-and-compounding', 90, 90, NOW() - INTERVAL '52 days'),
  ('sample-user-001', 'foundation', 'basics-of-risk', 96, 96, NOW() - INTERVAL '50 days'),
  ('sample-user-001', 'foundation', 'accounts-setup', 94, 94, NOW() - INTERVAL '48 days'),
  ('sample-user-001', 'foundation', 'first-investor-mindset', 91, 91, NOW() - INTERVAL '46 days'),
  ('sample-user-001', 'investor-insight', 'what-moves-markets', 97, 97, NOW() - INTERVAL '44 days'),
  ('sample-user-001', 'investor-insight', 'investor-psychology', 93, 93, NOW() - INTERVAL '42 days'),
  ('sample-user-001', 'investor-insight', 'hype-vs-fundamentals', 95, 95, NOW() - INTERVAL '40 days'),
  ('sample-user-001', 'investor-insight', 'types-of-investing', 92, 92, NOW() - INTERVAL '38 days'),
  ('sample-user-001', 'investor-insight', 'risk-portfolio', 94, 94, NOW() - INTERVAL '36 days'),
  ('sample-user-001', 'investor-insight', 'market-signals', 96, 96, NOW() - INTERVAL '34 days'),
  ('sample-user-001', 'applied-investing', 'costs-fees-taxes', 90, 90, NOW() - INTERVAL '32 days'),
  ('sample-user-001', 'applied-investing', 'market-crash', 93, 93, NOW() - INTERVAL '30 days'),
  ('sample-user-001', 'applied-investing', 'long-term-structure', 95, 95, NOW() - INTERVAL '28 days'),
  ('sample-user-001', 'applied-investing', 'realistic-returns', 91, 91, NOW() - INTERVAL '26 days'),
  ('sample-user-001', 'applied-investing', 'asset-allocation', 97, 97, NOW() - INTERVAL '24 days');

-- User 2: WealthBuilder - Strong performer
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-002', 'foundation', 'what-is-investing', 88, 88, NOW() - INTERVAL '48 days'),
  ('sample-user-002', 'foundation', 'stocks-bonds-funds', 92, 92, NOW() - INTERVAL '46 days'),
  ('sample-user-002', 'foundation', 'how-markets-work', 90, 90, NOW() - INTERVAL '44 days'),
  ('sample-user-002', 'foundation', 'time-and-compounding', 95, 95, NOW() - INTERVAL '42 days'),
  ('sample-user-002', 'foundation', 'basics-of-risk', 91, 91, NOW() - INTERVAL '40 days'),
  ('sample-user-002', 'foundation', 'accounts-setup', 89, 89, NOW() - INTERVAL '38 days'),
  ('sample-user-002', 'foundation', 'first-investor-mindset', 93, 93, NOW() - INTERVAL '36 days'),
  ('sample-user-002', 'investor-insight', 'what-moves-markets', 94, 94, NOW() - INTERVAL '34 days'),
  ('sample-user-002', 'investor-insight', 'investor-psychology', 90, 90, NOW() - INTERVAL '32 days'),
  ('sample-user-002', 'investor-insight', 'hype-vs-fundamentals', 92, 92, NOW() - INTERVAL '30 days'),
  ('sample-user-002', 'investor-insight', 'types-of-investing', 88, 88, NOW() - INTERVAL '28 days'),
  ('sample-user-002', 'investor-insight', 'risk-portfolio', 91, 91, NOW() - INTERVAL '26 days'),
  ('sample-user-002', 'investor-insight', 'market-signals', 93, 93, NOW() - INTERVAL '24 days'),
  ('sample-user-002', 'applied-investing', 'costs-fees-taxes', 89, 89, NOW() - INTERVAL '22 days'),
  ('sample-user-002', 'applied-investing', 'market-crash', 92, 92, NOW() - INTERVAL '20 days'),
  ('sample-user-002', 'applied-investing', 'long-term-structure', 90, 90, NOW() - INTERVAL '18 days'),
  ('sample-user-002', 'applied-investing', 'realistic-returns', 94, 94, NOW() - INTERVAL '16 days');

-- User 3: StockSavvy - Good performer
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-003', 'foundation', 'what-is-investing', 85, 85, NOW() - INTERVAL '53 days'),
  ('sample-user-003', 'foundation', 'stocks-bonds-funds', 90, 90, NOW() - INTERVAL '51 days'),
  ('sample-user-003', 'foundation', 'how-markets-work', 88, 88, NOW() - INTERVAL '49 days'),
  ('sample-user-003', 'foundation', 'time-and-compounding', 92, 92, NOW() - INTERVAL '47 days'),
  ('sample-user-003', 'foundation', 'basics-of-risk', 87, 87, NOW() - INTERVAL '45 days'),
  ('sample-user-003', 'foundation', 'accounts-setup', 91, 91, NOW() - INTERVAL '43 days'),
  ('sample-user-003', 'foundation', 'first-investor-mindset', 89, 89, NOW() - INTERVAL '41 days'),
  ('sample-user-003', 'investor-insight', 'what-moves-markets', 90, 90, NOW() - INTERVAL '39 days'),
  ('sample-user-003', 'investor-insight', 'investor-psychology', 86, 86, NOW() - INTERVAL '37 days'),
  ('sample-user-003', 'investor-insight', 'hype-vs-fundamentals', 93, 93, NOW() - INTERVAL '35 days'),
  ('sample-user-003', 'investor-insight', 'types-of-investing', 88, 88, NOW() - INTERVAL '33 days'),
  ('sample-user-003', 'investor-insight', 'risk-portfolio', 90, 90, NOW() - INTERVAL '31 days'),
  ('sample-user-003', 'investor-insight', 'market-signals', 87, 87, NOW() - INTERVAL '29 days'),
  ('sample-user-003', 'applied-investing', 'costs-fees-taxes', 91, 91, NOW() - INTERVAL '27 days'),
  ('sample-user-003', 'applied-investing', 'market-crash', 89, 89, NOW() - INTERVAL '25 days'),
  ('sample-user-003', 'applied-investing', 'long-term-structure', 92, 92, NOW() - INTERVAL '23 days');

-- User 4: MarketMaster - Solid performer
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-004', 'foundation', 'what-is-investing', 82, 82, NOW() - INTERVAL '38 days'),
  ('sample-user-004', 'foundation', 'stocks-bonds-funds', 88, 88, NOW() - INTERVAL '36 days'),
  ('sample-user-004', 'foundation', 'how-markets-work', 90, 90, NOW() - INTERVAL '34 days'),
  ('sample-user-004', 'foundation', 'time-and-compounding', 85, 85, NOW() - INTERVAL '32 days'),
  ('sample-user-004', 'foundation', 'basics-of-risk', 89, 89, NOW() - INTERVAL '30 days'),
  ('sample-user-004', 'foundation', 'accounts-setup', 87, 87, NOW() - INTERVAL '28 days'),
  ('sample-user-004', 'foundation', 'first-investor-mindset', 91, 91, NOW() - INTERVAL '26 days'),
  ('sample-user-004', 'investor-insight', 'what-moves-markets', 86, 86, NOW() - INTERVAL '24 days'),
  ('sample-user-004', 'investor-insight', 'investor-psychology', 88, 88, NOW() - INTERVAL '22 days'),
  ('sample-user-004', 'investor-insight', 'hype-vs-fundamentals', 84, 84, NOW() - INTERVAL '20 days'),
  ('sample-user-004', 'investor-insight', 'types-of-investing', 90, 90, NOW() - INTERVAL '18 days'),
  ('sample-user-004', 'investor-insight', 'risk-portfolio', 87, 87, NOW() - INTERVAL '16 days'),
  ('sample-user-004', 'investor-insight', 'market-signals', 85, 85, NOW() - INTERVAL '14 days'),
  ('sample-user-004', 'applied-investing', 'costs-fees-taxes', 89, 89, NOW() - INTERVAL '12 days'),
  ('sample-user-004', 'applied-investing', 'market-crash', 86, 86, NOW() - INTERVAL '10 days');

-- User 5: DividendKing - Good performer
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-005', 'foundation', 'what-is-investing', 80, 80, NOW() - INTERVAL '43 days'),
  ('sample-user-005', 'foundation', 'stocks-bonds-funds', 85, 85, NOW() - INTERVAL '41 days'),
  ('sample-user-005', 'foundation', 'how-markets-work', 88, 88, NOW() - INTERVAL '39 days'),
  ('sample-user-005', 'foundation', 'time-and-compounding', 90, 90, NOW() - INTERVAL '37 days'),
  ('sample-user-005', 'foundation', 'basics-of-risk', 83, 83, NOW() - INTERVAL '35 days'),
  ('sample-user-005', 'foundation', 'accounts-setup', 86, 86, NOW() - INTERVAL '33 days'),
  ('sample-user-005', 'foundation', 'first-investor-mindset', 84, 84, NOW() - INTERVAL '31 days'),
  ('sample-user-005', 'investor-insight', 'what-moves-markets', 87, 87, NOW() - INTERVAL '29 days'),
  ('sample-user-005', 'investor-insight', 'investor-psychology', 82, 82, NOW() - INTERVAL '27 days'),
  ('sample-user-005', 'investor-insight', 'hype-vs-fundamentals', 89, 89, NOW() - INTERVAL '25 days'),
  ('sample-user-005', 'investor-insight', 'types-of-investing', 85, 85, NOW() - INTERVAL '23 days'),
  ('sample-user-005', 'investor-insight', 'risk-portfolio', 88, 88, NOW() - INTERVAL '21 days'),
  ('sample-user-005', 'investor-insight', 'market-signals', 84, 84, NOW() - INTERVAL '19 days'),
  ('sample-user-005', 'applied-investing', 'costs-fees-taxes', 86, 86, NOW() - INTERVAL '17 days');

-- User 6: IndexFundFan - Average performer
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-006', 'foundation', 'what-is-investing', 78, 78, NOW() - INTERVAL '33 days'),
  ('sample-user-006', 'foundation', 'stocks-bonds-funds', 82, 82, NOW() - INTERVAL '31 days'),
  ('sample-user-006', 'foundation', 'how-markets-work', 85, 85, NOW() - INTERVAL '29 days'),
  ('sample-user-006', 'foundation', 'time-and-compounding', 80, 80, NOW() - INTERVAL '27 days'),
  ('sample-user-006', 'foundation', 'basics-of-risk', 84, 84, NOW() - INTERVAL '25 days'),
  ('sample-user-006', 'foundation', 'accounts-setup', 81, 81, NOW() - INTERVAL '23 days'),
  ('sample-user-006', 'foundation', 'first-investor-mindset', 86, 86, NOW() - INTERVAL '21 days'),
  ('sample-user-006', 'investor-insight', 'what-moves-markets', 79, 79, NOW() - INTERVAL '19 days'),
  ('sample-user-006', 'investor-insight', 'investor-psychology', 83, 83, NOW() - INTERVAL '17 days'),
  ('sample-user-006', 'investor-insight', 'hype-vs-fundamentals', 80, 80, NOW() - INTERVAL '15 days'),
  ('sample-user-006', 'investor-insight', 'types-of-investing', 82, 82, NOW() - INTERVAL '13 days'),
  ('sample-user-006', 'investor-insight', 'risk-portfolio', 85, 85, NOW() - INTERVAL '11 days'),
  ('sample-user-006', 'investor-insight', 'market-signals', 81, 81, NOW() - INTERVAL '9 days');

-- User 7: CompoundKing - Average performer
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-007', 'foundation', 'what-is-investing', 75, 75, NOW() - INTERVAL '28 days'),
  ('sample-user-007', 'foundation', 'stocks-bonds-funds', 80, 80, NOW() - INTERVAL '26 days'),
  ('sample-user-007', 'foundation', 'how-markets-work', 82, 82, NOW() - INTERVAL '24 days'),
  ('sample-user-007', 'foundation', 'time-and-compounding', 88, 88, NOW() - INTERVAL '22 days'),
  ('sample-user-007', 'foundation', 'basics-of-risk', 78, 78, NOW() - INTERVAL '20 days'),
  ('sample-user-007', 'foundation', 'accounts-setup', 81, 81, NOW() - INTERVAL '18 days'),
  ('sample-user-007', 'foundation', 'first-investor-mindset', 79, 79, NOW() - INTERVAL '16 days'),
  ('sample-user-007', 'investor-insight', 'what-moves-markets', 83, 83, NOW() - INTERVAL '14 days'),
  ('sample-user-007', 'investor-insight', 'investor-psychology', 77, 77, NOW() - INTERVAL '12 days'),
  ('sample-user-007', 'investor-insight', 'hype-vs-fundamentals', 84, 84, NOW() - INTERVAL '10 days'),
  ('sample-user-007', 'investor-insight', 'types-of-investing', 80, 80, NOW() - INTERVAL '8 days'),
  ('sample-user-007', 'investor-insight', 'risk-portfolio', 82, 82, NOW() - INTERVAL '6 days');

-- User 8: RetireEarly - Newer user
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-008', 'foundation', 'what-is-investing', 72, 72, NOW() - INTERVAL '23 days'),
  ('sample-user-008', 'foundation', 'stocks-bonds-funds', 78, 78, NOW() - INTERVAL '21 days'),
  ('sample-user-008', 'foundation', 'how-markets-work', 80, 80, NOW() - INTERVAL '19 days'),
  ('sample-user-008', 'foundation', 'time-and-compounding', 85, 85, NOW() - INTERVAL '17 days'),
  ('sample-user-008', 'foundation', 'basics-of-risk', 76, 76, NOW() - INTERVAL '15 days'),
  ('sample-user-008', 'foundation', 'accounts-setup', 79, 79, NOW() - INTERVAL '13 days'),
  ('sample-user-008', 'foundation', 'first-investor-mindset', 81, 81, NOW() - INTERVAL '11 days'),
  ('sample-user-008', 'investor-insight', 'what-moves-markets', 77, 77, NOW() - INTERVAL '9 days'),
  ('sample-user-008', 'investor-insight', 'investor-psychology', 82, 82, NOW() - INTERVAL '7 days'),
  ('sample-user-008', 'investor-insight', 'hype-vs-fundamentals', 78, 78, NOW() - INTERVAL '5 days'),
  ('sample-user-008', 'investor-insight', 'types-of-investing', 80, 80, NOW() - INTERVAL '3 days');

-- User 9: BudgetBoss - Newer user
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-009', 'foundation', 'what-is-investing', 70, 70, NOW() - INTERVAL '18 days'),
  ('sample-user-009', 'foundation', 'stocks-bonds-funds', 75, 75, NOW() - INTERVAL '16 days'),
  ('sample-user-009', 'foundation', 'how-markets-work', 78, 78, NOW() - INTERVAL '14 days'),
  ('sample-user-009', 'foundation', 'time-and-compounding', 82, 82, NOW() - INTERVAL '12 days'),
  ('sample-user-009', 'foundation', 'basics-of-risk', 74, 74, NOW() - INTERVAL '10 days'),
  ('sample-user-009', 'foundation', 'accounts-setup', 77, 77, NOW() - INTERVAL '8 days'),
  ('sample-user-009', 'foundation', 'first-investor-mindset', 79, 79, NOW() - INTERVAL '6 days'),
  ('sample-user-009', 'investor-insight', 'what-moves-markets', 76, 76, NOW() - INTERVAL '4 days'),
  ('sample-user-009', 'investor-insight', 'investor-psychology', 80, 80, NOW() - INTERVAL '2 days'),
  ('sample-user-009', 'investor-insight', 'hype-vs-fundamentals', 75, 75, NOW() - INTERVAL '1 day');

-- User 10: NewInvestor - Beginner
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-010', 'foundation', 'what-is-investing', 68, 68, NOW() - INTERVAL '13 days'),
  ('sample-user-010', 'foundation', 'stocks-bonds-funds', 72, 72, NOW() - INTERVAL '11 days'),
  ('sample-user-010', 'foundation', 'how-markets-work', 75, 75, NOW() - INTERVAL '9 days'),
  ('sample-user-010', 'foundation', 'time-and-compounding', 78, 78, NOW() - INTERVAL '7 days'),
  ('sample-user-010', 'foundation', 'basics-of-risk', 70, 70, NOW() - INTERVAL '5 days'),
  ('sample-user-010', 'foundation', 'accounts-setup', 74, 74, NOW() - INTERVAL '3 days'),
  ('sample-user-010', 'foundation', 'first-investor-mindset', 76, 76, NOW() - INTERVAL '1 day'),
  ('sample-user-010', 'investor-insight', 'what-moves-markets', 73, 73, NOW() - INTERVAL '12 hours');

-- User 11: SavingsGuru - Very new
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-011', 'foundation', 'what-is-investing', 65, 65, NOW() - INTERVAL '8 days'),
  ('sample-user-011', 'foundation', 'stocks-bonds-funds', 70, 70, NOW() - INTERVAL '6 days'),
  ('sample-user-011', 'foundation', 'how-markets-work', 72, 72, NOW() - INTERVAL '4 days'),
  ('sample-user-011', 'foundation', 'time-and-compounding', 75, 75, NOW() - INTERVAL '2 days'),
  ('sample-user-011', 'foundation', 'basics-of-risk', 68, 68, NOW() - INTERVAL '1 day'),
  ('sample-user-011', 'foundation', 'accounts-setup', 71, 71, NOW() - INTERVAL '6 hours');

-- User 12: FinanceFresh - Just started
INSERT INTO lesson_quiz_scores (user_id, module_id, lesson_id, score, percentage, created_at)
VALUES
  ('sample-user-012', 'foundation', 'what-is-investing', 60, 60, NOW() - INTERVAL '3 days'),
  ('sample-user-012', 'foundation', 'stocks-bonds-funds', 65, 65, NOW() - INTERVAL '1 day'),
  ('sample-user-012', 'foundation', 'how-markets-work', 68, 68, NOW() - INTERVAL '6 hours');

-- Insert module completion scores for top performers
INSERT INTO module_quiz_scores (user_id, module_id, score, percentage, created_at)
VALUES
  ('sample-user-001', 'foundation', 95, 95, NOW() - INTERVAL '44 days'),
  ('sample-user-001', 'investor-insight', 93, 93, NOW() - INTERVAL '32 days'),
  ('sample-user-001', 'applied-investing', 92, 92, NOW() - INTERVAL '22 days'),
  ('sample-user-002', 'foundation', 91, 91, NOW() - INTERVAL '34 days'),
  ('sample-user-002', 'investor-insight', 90, 90, NOW() - INTERVAL '22 days'),
  ('sample-user-002', 'applied-investing', 91, 91, NOW() - INTERVAL '14 days'),
  ('sample-user-003', 'foundation', 89, 89, NOW() - INTERVAL '39 days'),
  ('sample-user-003', 'investor-insight', 88, 88, NOW() - INTERVAL '27 days'),
  ('sample-user-003', 'applied-investing', 90, 90, NOW() - INTERVAL '21 days'),
  ('sample-user-004', 'foundation', 87, 87, NOW() - INTERVAL '24 days'),
  ('sample-user-004', 'investor-insight', 86, 86, NOW() - INTERVAL '12 days'),
  ('sample-user-005', 'foundation', 85, 85, NOW() - INTERVAL '29 days'),
  ('sample-user-005', 'investor-insight', 85, 85, NOW() - INTERVAL '17 days'),
  ('sample-user-006', 'foundation', 82, 82, NOW() - INTERVAL '19 days'),
  ('sample-user-006', 'investor-insight', 81, 81, NOW() - INTERVAL '7 days'),
  ('sample-user-007', 'foundation', 80, 80, NOW() - INTERVAL '14 days'),
  ('sample-user-007', 'investor-insight', 80, 80, NOW() - INTERVAL '4 days'),
  ('sample-user-008', 'foundation', 78, 78, NOW() - INTERVAL '9 days'),
  ('sample-user-009', 'foundation', 76, 76, NOW() - INTERVAL '4 days'),
  ('sample-user-010', 'foundation', 73, 73, NOW() - INTERVAL '1 day');

-- Verify the data
SELECT 
  p.display_name,
  COUNT(DISTINCT lqs.lesson_id) as lessons_completed,
  SUM(lqs.score) as total_lesson_score,
  ROUND(AVG(lqs.percentage), 1) as avg_percentage
FROM user_profiles p
LEFT JOIN lesson_quiz_scores lqs ON p.user_id = lqs.user_id
WHERE p.user_id LIKE 'sample-user-%'
GROUP BY p.user_id, p.display_name
ORDER BY total_lesson_score DESC;
