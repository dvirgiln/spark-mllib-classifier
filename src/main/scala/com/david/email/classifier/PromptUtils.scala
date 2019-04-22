package com.david.email.classifier


import scala.collection.immutable.NumericRange
import scala.io.StdIn
import scala.util.{Failure, Success, Try}

/*
 * Singleton object that is used to get the inputs from the console.
 */
object PromptUtils {

  def prompt(ln: Boolean = true) = {
    if (ln) println()
    print("prompt> ")
  }

  /*
   * Gets a range of Doubles from the user input
   */
  def getDoubleRange(message: String): Option[NumericRange.Inclusive[BigDecimal]] = {
    prompt(false)
    print(message)

    val input = StdIn.readLine()
    input match {
      case i if i.equals("") => None
      case _ =>
        val initRange = getDouble("Enter Range Initial Value(Double): ", "Range Initial Value is not a Double. Please retry.", false)
        val endRange = getDouble("Enter Range End Value(Double): ", "Range Initial Value is not a Double. Please retry.", false)
        val by = getDouble("Enter Range By Value(Double): ", "Range By Value is not a Double. Please retry.", false)
        Some((BigDecimal(initRange) to BigDecimal(endRange) by by))
    }
  }
  /*
   * Gets a range of Integers from the user input
   */
  def getIntRange(message: String): Option[Range] = {
    prompt(false)
    print(message)

    val input = StdIn.readLine()
    input match {
      case i if i.equals("") => None
      case _ =>
        val initRange = getInt("Enter Range Initial Value(Int): ", "Range Initial Value is not an Integer. Please retry.", false)
        val endRange = getInt("Enter Range End Value(Int): ", "Range Initial Value is not an Integer. Please retry.", false)
        val by = getInt("Enter Range By Value(Int): ", "Range By Value is not an Integer. Please retry.", false)
        Some((initRange to endRange by by))
    }
  }

  /*
 * Generic get that can be used to get an Int, Double, Char or any other type from the console input.
 * This is a good example of usage of higher order functions to simplify code.
 * @param message Message to prompt asking for a value
 * @param errorMessage message to display in case of exception.
 * @param isError boolean value to allow display the error message or not. First time, the value is false.
 * @param func contains the function that retrieves a value from user.
 */
  private def get[A](message: String, errorMessage: String, isError: Boolean, func: () => A): A = {
    prompt(false)
    isError match {
      case true =>
        print(errorMessage)
        prompt()
        print(message)
      case false => print(message)
    }
    Try(func.apply()) match {
      case Success(value) => value
      case Failure(exception) => get(message, errorMessage, true, func)
    }
  }

  /*
   * Function that gets a double from user input.
   */
  def getDouble(message: String, errorMessage: String, isError: Boolean): Double = get(message, errorMessage, isError, () => StdIn.readDouble)

  /*
   * Function that gets an int from user input.
   */
  def getInt(message: String, errorMessage: String, isError: Boolean): Int = get(message, errorMessage, isError, () => StdIn.readInt)
}
