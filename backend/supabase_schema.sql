-- =====================================================
-- Supabase Schema for FinLearn AI Learning System
-- Run this in Supabase SQL Editor
-- =====================================================

-- User Progress Table
CREATE TABLE IF NOT EXISTS user_progress (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  module_id TEXT NOT NULL,
  lesson_id TEXT NOT NULL,
  completed BOOLEAN DEFAULT FALSE,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, module_id, lesson_id)
);

-- Lesson Quiz Scores
CREATE TABLE IF NOT EXISTS lesson_quiz_scores (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  module_id TEXT NOT NULL,
  lesson_id TEXT NOT NULL,
  score INTEGER NOT NULL,
  total_questions INTEGER NOT NULL,
  percentage DECIMAL(5,2) NOT NULL,
  attempts INTEGER DEFAULT 1,
  best_score INTEGER,
  completed_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, module_id, lesson_id)
);

-- Module Final Quiz Scores (for leaderboard)
CREATE TABLE IF NOT EXISTS module_quiz_scores (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  module_id TEXT NOT NULL,
  score INTEGER NOT NULL,
  total_questions INTEGER NOT NULL,
  percentage DECIMAL(5,2) NOT NULL,
  attempts INTEGER DEFAULT 1,
  best_score INTEGER,
  best_percentage DECIMAL(5,2),
  time_taken_seconds INTEGER,
  completed_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, module_id)
);

-- User Profiles (for leaderboard display names)
CREATE TABLE IF NOT EXISTS user_profiles (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
  display_name TEXT,
  avatar_url TEXT,
  total_lessons_completed INTEGER DEFAULT 0,
  total_modules_completed INTEGER DEFAULT 0,
  total_quiz_points INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Leaderboard View (computed from module_quiz_scores)
CREATE OR REPLACE VIEW leaderboard AS
SELECT 
  up.user_id,
  COALESCE(up.display_name, SPLIT_PART(au.email, '@', 1)) as display_name,
  up.avatar_url,
  up.total_lessons_completed,
  up.total_modules_completed,
  COALESCE(SUM(mqs.best_score), 0) as total_quiz_score,
  COALESCE(AVG(mqs.best_percentage), 0) as avg_quiz_percentage,
  COUNT(DISTINCT mqs.module_id) as modules_quizzed
FROM user_profiles up
LEFT JOIN auth.users au ON up.user_id = au.id
LEFT JOIN module_quiz_scores mqs ON up.user_id = mqs.user_id
GROUP BY up.user_id, up.display_name, up.avatar_url, up.total_lessons_completed, up.total_modules_completed, au.email
ORDER BY total_quiz_score DESC, avg_quiz_percentage DESC;

-- Row Level Security Policies
ALTER TABLE user_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE lesson_quiz_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE module_quiz_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Users can read/write their own progress
CREATE POLICY "Users can manage own progress" ON user_progress
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own lesson scores" ON lesson_quiz_scores
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own module scores" ON module_quiz_scores
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own profile" ON user_profiles
  FOR ALL USING (auth.uid() = user_id);

-- Everyone can read leaderboard data (public profiles)
CREATE POLICY "Anyone can view leaderboard profiles" ON user_profiles
  FOR SELECT USING (true);

CREATE POLICY "Anyone can view module scores for leaderboard" ON module_quiz_scores
  FOR SELECT USING (true);

-- Function to update user profile stats
CREATE OR REPLACE FUNCTION update_user_stats()
RETURNS TRIGGER AS $$
BEGIN
  -- Update total lessons completed
  UPDATE user_profiles
  SET 
    total_lessons_completed = (
      SELECT COUNT(*) FROM user_progress 
      WHERE user_id = NEW.user_id AND completed = true
    ),
    updated_at = NOW()
  WHERE user_id = NEW.user_id;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update stats
CREATE TRIGGER on_progress_update
  AFTER INSERT OR UPDATE ON user_progress
  FOR EACH ROW
  EXECUTE FUNCTION update_user_stats();

-- Function to create user profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO user_profiles (user_id, display_name)
  VALUES (NEW.id, SPLIT_PART(NEW.email, '@', 1))
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for new user signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION handle_new_user();
