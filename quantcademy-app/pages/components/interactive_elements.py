"""
Interactive Elements for Learning Modules
Reusable components to make lessons engaging and personalized
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional

def inflation_calculator(initial_amount: int = 100, years: int = 10, default_inflation: float = 3.0):
    """
    Interactive inflation calculator showing purchasing power over time.
    
    Args:
        initial_amount: Starting amount in dollars
        years: Number of years to project
        default_inflation: Default inflation rate percentage
    """
    st.subheader("💰 Inflation Calculator: See How Your Money Loses Value")
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount ($)", min_value=1, value=initial_amount, step=10, key="infl_amount")
        years_input = st.slider("Years", 1, 50, years, key="infl_years")
    with col2:
        inflation_rate = st.slider("Inflation Rate (%)", 1.0, 8.0, default_inflation, 0.1, key="infl_rate")
    
    # Calculate future value
    future_value = amount / ((1 + inflation_rate/100) ** years_input)
    purchasing_power_loss = amount - future_value
    loss_percentage = (purchasing_power_loss / amount) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Value", f"${amount:,.2f}")
    with col2:
        st.metric("Future Purchasing Power", f"${future_value:,.2f}", 
                 delta=f"-{loss_percentage:.1f}%")
    with col3:
        st.metric("Value Lost", f"${purchasing_power_loss:,.2f}")
    
    # Visual chart
    year_range = np.arange(0, years_input + 1)
    values = [amount / ((1 + inflation_rate/100) ** y) for y in year_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_range,
        y=values,
        mode='lines+markers',
        name='Purchasing Power',
        line=dict(color='#ef4444', width=3),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)'
    ))
    fig.add_hline(y=amount, line_dash="dash", line_color="gray", 
                  annotation_text="Original Value", annotation_position="right")
    fig.update_layout(
        title=f"How ${amount:,} Loses Value Over {years_input} Years at {inflation_rate}% Inflation",
        xaxis_title="Years",
        yaxis_title="Purchasing Power ($)",
        height=350,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight
    st.info(f"💡 At {inflation_rate}% inflation, your ${amount:,} will only be worth ${future_value:,.2f} in {years_input} years. This is why investing matters!")


def saving_vs_investing_comparison(initial: int = 10000, monthly: int = 500, years: int = 20):
    """
    Interactive comparison tool showing savings vs investing over time.
    """
    st.subheader("💡 Saving vs Investing: See the Difference")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_amount = st.number_input("Starting Amount ($)", 0, 1000000, initial, 1000, key="comp_initial")
        monthly_contrib = st.number_input("Monthly Contribution ($)", 0, 10000, monthly, 50, key="comp_monthly")
    with col2:
        savings_rate = st.slider("Savings Account Rate (%)", 0.1, 3.0, 1.0, 0.1, key="comp_savings")
        years_input = st.slider("Time Horizon (years)", 5, 40, years, key="comp_years")
    with col3:
        investment_return = st.slider("Investment Return (%)", 4.0, 15.0, 7.0, 0.5, key="comp_invest")
    
    # Calculate both scenarios
    savings_total = initial_amount
    investment_total = initial_amount
    
    savings_history = [initial_amount]
    investment_history = [initial_amount]
    
    for year in range(years_input):
        savings_total = savings_total * (1 + savings_rate/100) + monthly_contrib * 12
        investment_total = investment_total * (1 + investment_return/100) + monthly_contrib * 12
        savings_history.append(savings_total)
        investment_history.append(investment_total)
    
    difference = investment_total - savings_total
    difference_percentage = (difference / savings_total) * 100
    
    # Display comparison
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Savings Account Total", f"${savings_total:,.0f}")
    with col2:
        st.metric("Investment Portfolio Total", f"${investment_total:,.0f}", 
                 delta=f"+${difference:,.0f} ({difference_percentage:.1f}% more)")
    
    # Visual comparison chart
    year_range = list(range(years_input + 1))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=year_range,
        y=savings_history,
        mode='lines+markers',
        name=f'Savings ({savings_rate}%)',
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=year_range,
        y=investment_history,
        mode='lines+markers',
        name=f'Investing ({investment_return}%)',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    fig.update_layout(
        title=f"Growth Comparison Over {years_input} Years",
        xaxis_title="Years",
        yaxis_title="Total Value ($)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insight
    if difference > 0:
        st.success(f"🎯 Investing would give you **${difference:,.0f} more** over {years_input} years! That's {difference_percentage:.1f}% more than saving.")
    else:
        st.warning("⚠️ With these settings, savings performs better. Consider adjusting the investment return rate.")


def compound_interest_calculator(principal: int = 10000, monthly: int = 500, rate: float = 7.0, years: int = 30):
    """
    Interactive compound interest calculator.
    """
    st.subheader("🚀 Compound Interest Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        initial = st.number_input("Initial Investment ($)", 0, 1000000, principal, 1000, key="comp_int_initial")
        monthly_contrib = st.number_input("Monthly Contribution ($)", 0, 10000, monthly, 50, key="comp_int_monthly")
    with col2:
        annual_return = st.slider("Annual Return (%)", 3.0, 15.0, rate, 0.5, key="comp_int_rate")
        years_input = st.slider("Years", 1, 50, years, key="comp_int_years")
    
    # Calculate compound interest
    monthly_rate = annual_return / 12 / 100
    total_months = years_input * 12
    
    # Future value of initial investment
    fv_initial = initial * ((1 + monthly_rate) ** total_months)
    
    # Future value of monthly contributions (annuity)
    if monthly_rate > 0:
        fv_annuity = monthly_contrib * (((1 + monthly_rate) ** total_months - 1) / monthly_rate)
    else:
        fv_annuity = monthly_contrib * total_months
    
    total_future_value = fv_initial + fv_annuity
    total_contributed = initial + (monthly_contrib * total_months)
    total_growth = total_future_value - total_contributed
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Contributed", f"${total_contributed:,.0f}")
    with col2:
        st.metric("Future Value", f"${total_future_value:,.0f}")
    with col3:
        st.metric("Growth", f"${total_growth:,.0f}", 
                 delta=f"{(total_growth/total_contributed)*100:.1f}%")
    
    # Growth over time chart
    months = list(range(0, total_months + 1, 12))  # Yearly points
    values = []
    contributed = []
    
    for year in range(years_input + 1):
        months_elapsed = year * 12
        if months_elapsed == 0:
            val = initial
            contrib = initial
        else:
            val = initial * ((1 + monthly_rate) ** months_elapsed)
            if monthly_rate > 0:
                val += monthly_contrib * (((1 + monthly_rate) ** months_elapsed - 1) / monthly_rate)
            else:
                val += monthly_contrib * months_elapsed
            contrib = initial + (monthly_contrib * months_elapsed)
        values.append(val)
        contributed.append(contrib)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(years_input + 1)),
        y=contributed,
        mode='lines',
        name='Total Contributed',
        line=dict(color='#6b7280', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(years_input + 1)),
        y=values,
        mode='lines+markers',
        name='Future Value',
        line=dict(color='#10b981', width=3),
        fill='tonexty',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    fig.update_layout(
        title=f"Growth Over {years_input} Years at {annual_return}% Annual Return",
        xaxis_title="Years",
        yaxis_title="Value ($)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insight
    st.info(f"💡 Your ${total_contributed:,.0f} would grow to ${total_future_value:,.0f} - that's **${total_growth:,.0f} in growth** thanks to compound interest!")


def concept_check(question: str, options: list, correct_index: int, explanation: str = ""):
    """
    Interactive concept check with immediate feedback.
    """
    st.subheader("🧠 Quick Check: Do You Understand?")
    
    answer = st.radio(question, options, key=f"concept_{hash(question)}")
    
    if st.button("Check Answer", key=f"check_{hash(question)}"):
        if answer == options[correct_index]:
            st.success("✅ Correct! Great job understanding this concept.")
            if explanation:
                st.info(f"💡 {explanation}")
            st.balloons()
        else:
            st.error("❌ Not quite. Let's review this concept.")
            st.info(f"💡 The correct answer is: **{options[correct_index]}**")
            if explanation:
                st.info(explanation)
    
    return answer == options[correct_index] if 'answer' in locals() else None


def ai_tutor_sidebar(lesson_id: str, lesson_title: str):
    """
    AI Tutor sidebar for lesson-specific questions.
    """
    with st.sidebar:
        st.subheader("🤖 Ask the AI Tutor")
        st.caption(f"Get personalized answers about: {lesson_title}")
        
        user_question = st.text_input(
            "Ask a question...", 
            placeholder="e.g., How does this apply to my situation?",
            key=f"tutor_q_{lesson_id}"
        )
        
        if user_question and st.button("Ask", key=f"tutor_btn_{lesson_id}"):
            try:
                from rag.retrieval import retrieve_with_citations, format_context_with_citations
                from rag.llm_provider import chat_with_llm, check_llm_status
                
                # Check if LLM is available
                llm_status = check_llm_status()
                if llm_status.get('status') != 'online':
                    st.warning("⚠️ LLM not configured. Please set up Gemini or Ollama in your .env file.")
                    st.caption(f"The AI tutor needs an LLM to synthesize answers. Status: {llm_status.get('message', 'Unknown')}")
                    return
                
                with st.spinner("Retrieving relevant information..."):
                    # Step 1: Retrieve relevant chunks (top 5 after reranking)
                    response = retrieve_with_citations(
                        user_question,
                        current_lesson_id=lesson_id,
                        top_k=5
                    )
                
                if not response.is_confident:
                    st.warning("⚠️ I'm not confident about this answer.")
                    if response.refusal_reason:
                        st.caption(response.refusal_reason)
                    st.info("💡 Try asking about concepts from this lesson or rephrasing your question.")
                    return
                
                if not response.results:
                    st.warning("No relevant information found. Try asking about concepts from this lesson.")
                    return
                
                # Step 2: Format context from retrieved chunks
                context, citations_str = format_context_with_citations(response)
                
                # Step 3: Synthesize answer using LLM
                with st.spinner("Synthesizing answer..."):
                    st.markdown("### Answer")
                    
                    # Stream the LLM response
                    answer_placeholder = st.empty()
                    full_answer = ""
                    
                    try:
                        for chunk in chat_with_llm(
                            message=user_question,
                            context=context,
                            citations=citations_str,
                            confidence=response.confidence,
                            is_confident=response.is_confident,
                            refusal_reason=response.refusal_reason,
                            stream=True
                        ):
                            full_answer += chunk
                            answer_placeholder.markdown(full_answer)
                        
                        # Store the full answer in session state for potential follow-up
                        st.session_state[f"tutor_answer_{lesson_id}"] = full_answer
                        
                    except Exception as llm_error:
                        st.error(f"Error generating answer: {str(llm_error)}")
                        # Fallback: show raw context if LLM fails
                        st.info("Showing retrieved information:")
                        st.markdown(context[:1000])
                        if len(context) > 1000:
                            st.caption("... (truncated)")
                
                # Show sources
                if response.citations:
                    st.markdown("---")
                    st.caption(f"📚 **Sources:** {', '.join(response.citations[:3])}")
                    
            except ImportError as e:
                st.error(f"Import error: {str(e)}")
                st.caption("Make sure the RAG system is properly configured.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.caption("Make sure the RAG system and LLM are properly configured.")


# Example usage in a lesson:
"""
# In your lesson content, you can now use:

from pages.components.interactive_elements import (
    inflation_calculator,
    saving_vs_investing_comparison,
    compound_interest_calculator,
    concept_check,
    ai_tutor_sidebar
)

# Add interactive elements
inflation_calculator(initial_amount=100, years=10)

saving_vs_investing_comparison(initial=10000, monthly=500, years=20)

concept_check(
    question="Which is better for a 20-year retirement goal?",
    options=["Savings account (1% return)", "Investment portfolio (7% return)", "Not sure"],
    correct_index=1,
    explanation="For long-term goals, investing typically provides better returns due to compound growth."
)

ai_tutor_sidebar(lesson_id="what_is_investing", lesson_title="What is Investing?")
"""
