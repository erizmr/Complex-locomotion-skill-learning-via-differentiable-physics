

if __name__ == "__main__":
    args = get_args()
    ConfigParser.from_args(args, options)
    # config =
    trainer = RLTrainer(args, config=config)
    trainer.train(0, 1000)